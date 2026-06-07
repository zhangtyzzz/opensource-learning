# 08 - Agent 系统：从链式调用到图状态机

## 一次彻底的架构转向

LangChain v1 中最引人注目的变化不是新增了什么功能，而是**删除了什么**。曾经作为框架核心卖点的 `AgentExecutor`、`ZeroShotAgent`、`ConversationalAgent` 等 Agent 抽象全部被标记为 deprecated，冻结在 `langchain-classic` 包中。取而代之的是一个极其精简的新模块：整个 `langchain_v1/langchain/agents/` 目录只有 4 个文件——`factory.py`、`_subagent_transformer.py`、`structured_output.py` 和一个 middleware 子目录。

这不是重构，而是投降——LangChain 放弃了自己构建 Agent 运行时的尝试，将这一职责完全交给了 LangGraph。新版 `langchain` 包的 `pyproject.toml` 中明确声明了对 `langgraph>=1.2.4` 的硬依赖。`create_agent` 工厂函数本质上是 LangGraph `StateGraph` 之上的一层高级封装。

```
旧架构（langchain-classic）                新架构（langchain v1）
+---------------------------+           +---------------------------+
| AgentExecutor             |           | create_agent()            |
|   while not finished:     |           |   StateGraph 构建         |
|     action = agent(...)   |           |   middleware 注册          |
|     result = tool(action) |           |   compile() -> 图执行      |
|     agent.memory.save()   |           +---------------------------+
+---------------------------+                      |
    自研循环 + 自研内存                          依赖 LangGraph
                                          (checkpointer, interrupt,
                                           streaming, state channels)
```

## create_agent 工厂函数

`create_agent` 是新 Agent 系统的唯一入口。它的签名揭示了设计意图——用最少的参数创建一个功能完整的 Agent 图：

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",  # 字符串或 BaseChatModel 实例
    tools=[search_tool, calculator],                # 工具列表
    system_prompt="You are a helpful assistant",    # 系统提示
    middleware=[retry, hitl],                        # 中间件序列
    response_format=MySchema,                       # 可选结构化输出
    checkpointer=memory,                            # LangGraph checkpointer
)
```

在内部，`create_agent` 执行以下步骤（源码位于 `libs/langchain_v1/langchain/agents/factory.py`，共 1,892 行）：

1. **模型初始化**：字符串模型标识（如 `"openai:gpt-5"`）通过 `init_chat_model` 转换为 `BaseChatModel` 实例
2. **工具注册**：收集用户工具 + 中间件附带的工具，创建 LangGraph `ToolNode`
3. **中间件编排**：按钩子类型分类中间件（before_agent / before_model / wrap_model_call / after_model / after_agent / wrap_tool_call），组合为中间件栈
4. **状态 Schema 合并**：将 `AgentState`、用户自定义 `state_schema` 和所有中间件的 `state_schema` 合并为统一的 TypedDict
5. **图构建**：创建 `StateGraph`，添加 model 节点、tools 节点和中间件节点，连接条件边
6. **编译**：调用 `graph.compile()` 生成 `CompiledStateGraph`，注册 `ToolCallTransformer` 和 `SubagentTransformer`

构建出的图结构如下：

```
START
  |
  v
[before_agent middleware(s)]  <-- 仅在首次进入时执行
  |
  v
[before_model middleware(s)]  <-- 循环入口
  |
  v
[model node]  -----------------> 调用 LLM
  |
  v
[after_model middleware(s)]
  |
  |--- 有 tool_calls? --yes--> [tools node] --> 回到 before_model（循环）
  |--- 无 tool_calls? --no---> [after_agent middleware(s)] --> END
```

## AgentState：图状态的类型契约

`AgentState` 是一个 `TypedDict`，定义了图中所有节点共享的状态结构：

```python
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

三个字段各有设计考量：

- **`messages`**：使用 `add_messages` 作为 reducer，这意味着新消息是追加而非覆盖。这是从 LangGraph 的 channel 系统继承的关键语义——多个节点可以独立向消息列表写入，reducer 负责合并。
- **`jump_to`**：标记为 `EphemeralValue`（短暂值），在一个步骤后自动清除。中间件通过设置 `jump_to` 来改变图的执行流向——跳到 `"model"`、`"tools"` 或 `"end"`。
- **`structured_response`**：标记为 `OmitFromInput`（不出现在输入 schema 中），仅在使用 `response_format` 时由 model 节点写入。

中间件可以通过 `state_schema` 类属性扩展状态。例如 `SummarizationMiddleware` 需要跟踪摘要历史，它可以向状态中注入额外字段。`create_agent` 在构建图时会将所有 schema 合并为一个统一的 TypedDict（通过 `_resolve_schemas` 函数）。

## Middleware 中间件架构

中间件是 v1 Agent 系统的核心扩展机制。`AgentMiddleware` 基类定义了六个钩子点：

```python
class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    # 生命周期钩子（作为图节点执行）
    def before_agent(self, state, runtime): ...   # Agent 启动前（仅一次）
    def before_model(self, state, runtime): ...   # 每次模型调用前
    def after_model(self, state, runtime): ...    # 每次模型调用后
    def after_agent(self, state, runtime): ...    # Agent 结束后（仅一次）

    # 包装器钩子（拦截实际执行）
    def wrap_model_call(self, request, handler): ...  # 包装模型调用
    def wrap_tool_call(self, request, handler): ...   # 包装工具调用
```

生命周期钩子（before/after）被编译为图中的独立节点；包装器钩子（wrap）采用洋葱模型组合——第一个中间件在最外层，最后一个在最内层：

```
wrap_model_call 组合顺序（middleware=[A, B, C]）：

请求 --> A.wrap --> B.wrap --> C.wrap --> 实际模型调用
响应 <-- A.wrap <-- B.wrap <-- C.wrap <-- 模型响应
```

组合通过 `_chain_model_call_handlers` 函数实现，它将多个 handler 右折叠为一个嵌套的调用链。每个中间件收到的 `handler` 参数实际上是"剩余中间件 + 真实执行"的组合体。

### 内置中间件一览

| 中间件 | 钩子类型 | 功能 |
|--------|----------|------|
| `ModelRetryMiddleware` | `wrap_model_call` | 指数退避重试，可配置重试条件和最大次数 |
| `ModelFallbackMiddleware` | `wrap_model_call` | 主模型失败时按序尝试备选模型 |
| `HumanInTheLoopMiddleware` | `after_model` | 在工具调用前触发 `interrupt()`，等待人工审批/编辑/拒绝 |
| `PIIMiddleware` | `before_model` + stream transformer | 检测并脱敏 PII（邮箱、信用卡、IP 等），包括流式输出的实时脱敏 |
| `SummarizationMiddleware` | `wrap_model_call` | 当对话超过 token 阈值时，用 LLM 生成摘要替换旧消息 |
| `ContextEditingMiddleware` | `wrap_model_call` | 超过 token 阈值时清除旧的工具调用结果，释放上下文空间 |
| `LLMToolSelectorMiddleware` | `wrap_model_call` | 用一个轻量 LLM 先筛选相关工具，减少主模型的工具列表 |
| `ToolCallLimitMiddleware` | `wrap_tool_call` | 限制单次运行中的工具调用总次数 |
| `ModelCallLimitMiddleware` | `wrap_model_call` | 限制模型调用总次数 |
| `ShellToolMiddleware` | 注册工具 | 提供安全的 shell 命令执行工具（支持 Docker/Codex 沙箱） |

### ModelRetryMiddleware 深度示例

`ModelRetryMiddleware` 展示了 `wrap_model_call` 模式的典型用法：

```python
class ModelRetryMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)  # 调用下一层（可能是另一个中间件或真实执行）
            except Exception as exc:
                if not should_retry_exception(exc, self.retry_on):
                    return self._handle_failure(exc, attempt + 1)
                if attempt < self.max_retries:
                    delay = calculate_delay(attempt, ...)
                    time.sleep(delay)
                else:
                    return self._handle_failure(exc, attempt + 1)
```

关键设计：失败时不抛异常，而是返回一个包含错误信息的 `AIMessage`，让 Agent 循环可以继续运行。这与 `on_failure="error"` 模式（直接抛异常终止图执行）形成对比，开发者可以根据场景选择。

### HumanInTheLoopMiddleware：图中断的封装

人工审核中间件是 LangGraph `interrupt()` 机制在 LangChain Agent 层面的封装。它作为 `after_model` 钩子运行：

```python
# 配置哪些工具需要人工审批
hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_file": True,                         # 所有决策类型
        "send_email": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],  # 只允许批准或拒绝
            when=lambda req: req.tool_call["args"].get("to") != "self",  # 条件触发
        ),
    }
)
```

当模型生成的 `AIMessage` 包含需要审批的 tool_calls 时，中间件构造 `HITLRequest`，调用 `interrupt()` 暂停图执行。人工决策返回后，中间件根据决策类型（approve / edit / reject / respond）修改或过滤 tool_calls，然后让图继续执行。

### 装饰器快捷方式

除了继承 `AgentMiddleware` 类，还可以用装饰器快速创建中间件：

```python
from langchain.agents.middleware import before_model, wrap_model_call

@before_model
def log_messages(state: AgentState, runtime: Runtime) -> None:
    print(f"Calling model with {len(state['messages'])} messages")

@wrap_model_call
def add_caching(request, handler):
    if cached := lookup_cache(request.messages):
        return cached
    response = handler(request)
    save_cache(request.messages, response)
    return response

agent = create_agent(model, tools=[...], middleware=[log_messages, add_caching])
```

装饰器内部动态创建一个 `AgentMiddleware` 子类并实例化——本质上是类继承的语法糖。

## SubagentTransformer：子 Agent 委派

`SubagentTransformer`（源码位于 `_subagent_transformer.py`）处理嵌套 Agent 场景——当一个 Agent 的工具内部调用了另一个 `create_agent(name="sub_agent")` 创建的 Agent 时，SubagentTransformer 会：

1. 检测命名空间中 `lc_agent_name` 的变化（子 Agent 的名字与父 Agent 不同）
2. 为子 Agent 创建独立的 `StreamMux`（流复用器）
3. 在 `run.subagents` 上暴露类型化的 `SubagentRunStream` 句柄
4. 将子命名空间的事件转发到子 Agent 的 mux 中

这使得消费者可以独立订阅和处理子 Agent 的执行流，而不是将所有事件混在一起。每个子 Agent 句柄还携带 `cause` 属性（触发它的 tool_call_id），保持了因果关系的可追溯性。

## 从 AgentExecutor 到 LangGraph 的迁移

旧式 `AgentExecutor` 的核心是一个 Python while 循环：

```python
# langchain-classic 中的 AgentExecutor（简化）
class AgentExecutor:
    def _call(self, inputs):
        while True:
            action = self.agent.plan(intermediate_steps, **inputs)
            if isinstance(action, AgentFinish):
                return action.return_values
            observation = self._take_action(action)
            intermediate_steps.append((action, observation))
```

这种设计的问题在于：所有状态管理、错误处理、人工中断、流式输出、持久化都需要在这个循环内自行实现。而 LangGraph 将这些能力作为图运行时的基础设施提供：

| 能力 | AgentExecutor | create_agent (LangGraph) |
|------|---------------|--------------------------|
| 状态持久化 | 手动实现 Memory | Checkpointer（自动） |
| 人工中断 | 不支持 | `interrupt()` + checkpoint |
| 流式输出 | Callback hack | StreamMux + 事件协议 |
| 错误恢复 | 自行 try/catch | Middleware + 图级重试 |
| 并行工具调用 | 不支持 | `Send` API 并行分发 |
| 子 Agent | 嵌套 AgentExecutor | SubagentTransformer |

## 设计思考

### 为什么放弃自研 Agent 抽象

LangChain 放弃自己的 Agent 运行时，转而依赖 LangGraph，这是一个值得深思的架构决策。

**表面原因是避免重复造轮子。** LangGraph 已经解决了状态管理、持久化、中断恢复、流式处理等图执行的核心难题。在 `AgentExecutor` 内部重新实现这些能力意味着维护两套运行时。

**深层原因是抽象层级的认知。** Agent 本质上是一个有条件循环的状态机——"调用模型，如果有工具调用则执行工具，否则结束"。这恰好是 LangGraph 的 StateGraph 原语可以直接表达的。LangChain 团队意识到，Agent 不需要自己的运行时——它需要的是一个好的图运行时之上的**模式**（pattern），而不是另一个**框架**（framework）。

**这重新定义了 LangChain 的角色。** 在 v1 中，`langchain` 包变成了一个薄薄的"体验层"——它不拥有任何执行基础设施，只提供 `create_agent`（图构建模式）、`init_chat_model`（模型初始化便捷方式）和中间件（可复用的 Agent 行为模块）。真正的重型工作全部委托给 `langchain-core`（类型系统）和 `langgraph`（执行引擎）。

### Middleware 模式的得与失

中间件系统的设计是精巧的——将 Agent 行为分解为可组合的横切关注点（重试、审批、脱敏等），这是一个成熟的软件工程模式。但它也引入了复杂性：

1. **组合顺序敏感**：`[retry, fallback]` 和 `[fallback, retry]` 行为完全不同。前者先重试再降级，后者先降级再重试。文档需要非常清楚地解释洋葱模型。
2. **状态 schema 合并**：当多个中间件各自扩展状态时，字段冲突的解决规则不够透明（后定义的覆盖先定义的）。
3. **同步/异步双轨制**：每个钩子都有 sync 和 async 两个版本（`wrap_model_call` / `awrap_model_call`），如果只实现了一个版本而在另一个上下文中调用，会得到 `NotImplementedError`。

### 框架边界的重新定义

v1 的 Agent 架构暗示了一个更大的问题：**LangChain 到底是什么？** 它不再是一个编排框架（那是 LangGraph），不再是一个模型接口层（那是 `langchain-core`），不再是一个集成库（那是 partner packages）。它变成了一个**粘合层**——用最少的代码将 LangGraph 的图能力、langchain-core 的类型系统和 partner 集成的模型实现连接成一个开箱即用的 Agent 体验。

这种定位是务实的，但也是脆弱的。如果开发者直接使用 LangGraph 的 `StateGraph` 构建 Agent（这完全可行且更灵活），`create_agent` 和它的中间件系统就变得可有可无。LangChain v1 的 Agent 模块本质上是在赌一个判断：大多数开发者更想要开箱即用的 Agent 模式，而不是从零搭建图。

> **下一章**: [09 - 可观测性：回调、追踪与事件流](09-observability-and-tracing.md)