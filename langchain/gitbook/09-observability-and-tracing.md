# 09 - 可观测性：回调、追踪与事件流

LLM 应用天然是黑盒：一个 Agent 执行可能涉及多轮模型调用、工具执行、检索查询，整个过程对开发者不透明。可观测性不是锦上添花，而是生产环境的基础设施。LangChain 为此构建了一套双层可观测体系——底层的 Callback 系统提供生命周期钩子，上层的 Tracer 系统在回调基础上实现结构化追踪。

## 总体架构

```
+------------------------------------------------------------------+
|                       用户代码                                     |
|   chain.invoke(input, config={"callbacks": [handler]})           |
+------------------------------------------------------------------+
         |                                    |
         v                                    v
+-------------------+              +---------------------+
|  CallbackManager  |              |   RunnableConfig    |
|  (事件分发中枢)     |  <-------->  |  callbacks / tags / |
|                   |    自动传播    |  metadata           |
+-------------------+              +---------------------+
         |
         | 广播事件到所有注册的 handler
         v
+------------------------------------------------------------------+
|                    BaseCallbackHandler                             |
|  +-----------------+  +----------------+  +--------------------+  |
|  | StdOutCallback  |  | LangChainTracer|  | UsageMetadata      |  |
|  | Handler         |  | (-> LangSmith) |  | CallbackHandler    |  |
|  +-----------------+  +----------------+  +--------------------+  |
|  +-------------------+  +------------------+                      |
|  | EventStream      |  | LogStream        |                      |
|  | CallbackHandler  |  | CallbackHandler  |                      |
|  +-------------------+  +------------------+                      |
+------------------------------------------------------------------+
```

**源码路径：**
- 回调基础：`libs/core/langchain_core/callbacks/base.py`
- 回调管理：`libs/core/langchain_core/callbacks/manager.py`
- 追踪核心：`libs/core/langchain_core/tracers/core.py`
- LangSmith 追踪：`libs/core/langchain_core/tracers/langchain.py`
- 事件流：`libs/core/langchain_core/tracers/event_stream.py`

## 第一层：Callback 系统

### BaseCallbackHandler 的 Mixin 体系

LangChain 的回调处理器不是一个单一的大接口，而是通过 Mixin 模式组合而成。每种组件类型拥有独立的 Mixin：

```python
class BaseCallbackHandler(
    LLMManagerMixin,         # on_llm_new_token / on_llm_end / on_llm_error
    ChainManagerMixin,       # on_chain_end / on_chain_error / on_agent_action
    ToolManagerMixin,        # on_tool_end / on_tool_error
    RetrieverManagerMixin,   # on_retriever_end / on_retriever_error
    CallbackManagerMixin,    # on_llm_start / on_chat_model_start / on_chain_start
                             # on_tool_start / on_retriever_start
    RunManagerMixin,         # on_text / on_retry / on_custom_event
):
    raise_error: bool = False   # 异常是否向上冒泡
    run_inline: bool = False    # 是否在当前线程内联执行
```

注意 `start` 钩子和 `end/error` 钩子分布在不同的 Mixin 中——`CallbackManagerMixin` 包含所有 `start` 事件，而各组件的 Manager Mixin 包含对应的 `end` 和 `error` 事件。这个拆分反映了生命周期的语义差异：`start` 事件由 `CallbackManager` 在外层触发（需要创建子 RunManager），而 `end/error` 事件由具体的 `RunManager` 在运行结束时触发。

### 完整的生命周期钩子

| 组件类型 | start | 中间事件 | end | error |
|---------|-------|---------|-----|-------|
| LLM | `on_llm_start` | `on_llm_new_token` | `on_llm_end` | `on_llm_error` |
| ChatModel | `on_chat_model_start` | `on_llm_new_token` / `on_stream_event` | `on_llm_end` | `on_llm_error` |
| Chain | `on_chain_start` | - | `on_chain_end` | `on_chain_error` |
| Tool | `on_tool_start` | - | `on_tool_end` | `on_tool_error` |
| Retriever | `on_retriever_start` | - | `on_retriever_end` | `on_retriever_error` |
| Agent | - | `on_agent_action` / `on_agent_finish` | - | - |
| 通用 | - | `on_text` / `on_retry` / `on_custom_event` | - | - |

一个关键的兼容性设计：`on_chat_model_start` 默认抛出 `NotImplementedError`。当 `CallbackManager` 检测到这个异常时，会自动回退到 `on_llm_start`，将消息序列化为字符串传入。这允许旧的 handler 无需修改就能处理 ChatModel 事件：

```python
# 在 handle_event 函数中（manager.py）
except NotImplementedError as e:
    if event_name == "on_chat_model_start":
        message_strings = [get_buffer_string(m) for m in args[1]]
        handle_event([handler], "on_llm_start", "ignore_llm",
                     args[0], message_strings, ...)
```

### ignore_* 过滤机制

每个 handler 可以声明忽略特定类型的事件：

```python
class MyHandler(BaseCallbackHandler):
    @property
    def ignore_llm(self) -> bool:
        return True  # 不接收 LLM 相关事件

    @property
    def ignore_chain(self) -> bool:
        return True  # 不接收 Chain 相关事件
```

`CallbackManager` 在分发事件时，会检查 `ignore_condition_name` 属性决定是否跳过该 handler。可用的过滤属性包括：`ignore_llm`、`ignore_chain`、`ignore_agent`、`ignore_retriever`、`ignore_chat_model`、`ignore_retry`、`ignore_custom_event`。

### 同步/异步事件处理

`CallbackManager` 对同步和异步 handler 的处理策略不同，而且需要处理一个棘手的场景——在同步代码中使用异步 handler：

```python
# handle_event（同步入口）
def handle_event(handlers, event_name, ignore_condition_name, *args, **kwargs):
    coros = []
    for handler in handlers:
        event = getattr(handler, event_name)(*args, **kwargs)
        if asyncio.iscoroutine(event):
            coros.append(event)  # 收集异步协程

    if coros:
        if loop_running:
            # 死锁规避：不能在运行中的事件循环里直接 await
            # 提交到全局线程池的独立事件循环中执行
            _executor().submit(copy_context().run, _run_coros, coros).result()
        else:
            _run_coros(coros)  # 无事件循环，直接运行
```

这个全局线程池（`ThreadPoolExecutor(max_workers=10)`）通过 `@functools.lru_cache(maxsize=1)` 惰性创建，程序退出时通过 `atexit` 注册清理。

在异步路径中，`run_inline` 标记的 handler 串行执行，其余 handler 通过 `asyncio.gather` 并行执行：

```python
async def ahandle_event(handlers, event_name, ...):
    # 先执行 run_inline handler（串行）
    for handler in [h for h in handlers if h.run_inline]:
        await _ahandle_event_for_handler(handler, ...)
    # 再并行执行其余 handler
    await asyncio.gather(*(
        _ahandle_event_for_handler(h, ...)
        for h in handlers if not h.run_inline
    ))
```

## CallbackManager 的传播机制

### inheritable 与 local 的双轨设计

`BaseCallbackManager` 维护两组数据：

```python
class BaseCallbackManager:
    handlers: list[BaseCallbackHandler]              # 当前层的所有 handler
    inheritable_handlers: list[BaseCallbackHandler]   # 会传递给子组件的 handler
    tags: list[str]
    inheritable_tags: list[str]
    metadata: dict[str, Any]
    inheritable_metadata: dict[str, Any]
```

当一个 Runnable 执行内部子组件时，通过 `ParentRunManager.get_child()` 创建子 `CallbackManager`——只继承 `inheritable_*` 部分：

```python
class ParentRunManager(RunManager):
    def get_child(self, tag=None) -> CallbackManager:
        manager = CallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)  # 只传播可继承的
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        return manager
```

这个设计的实际效果：通过 `add_handler(handler, inherit=False)` 添加的 handler 只作用于当前层级，不会传递到子调用。例如 `StdOutCallbackHandler` 在 verbose 模式下以 `inherit=False` 添加，避免子组件重复输出。

### RunnableConfig 中的 callbacks 传播

Callback 的自动传播通过 `RunnableConfig` 和 `CallbackManager.configure()` 实现。每个 Runnable 在执行时：

1. 从 `RunnableConfig` 中取出 `callbacks` 字段
2. 调用 `CallbackManager.configure()` 合并 inheritable 和 local 回调
3. 触发 `on_*_start` 事件，获得绑定了 `run_id` 的 `RunManager`
4. 将 `RunManager` 传递给实际执行逻辑
5. 执行结束后通过 `RunManager` 触发 `on_*_end` 或 `on_*_error`

`configure()` 方法还会检查环境变量自动注入 handler：

```python
# _configure() 函数核心逻辑
if tracing_v2_enabled_:
    # LANGCHAIN_TRACING_V2=true 时自动注入 LangChainTracer
    callback_manager.add_handler(LangChainTracer(project_name=...))

if debug:
    # langchain.debug = True 时注入 ConsoleCallbackHandler
    callback_manager.add_handler(ConsoleCallbackHandler())

if verbose:
    # verbose=True 时注入 StdOutCallbackHandler（不可继承）
    callback_manager.add_handler(StdOutCallbackHandler(), inherit=False)
```

### ContextVar 注册钩子

`register_configure_hook` 提供了一种将 `ContextVar` 中的 handler 自动注入到 `CallbackManager` 的机制：

```python
# tracers/context.py
_configure_hooks: list[tuple[ContextVar, bool, type | None, str | None]] = []

def register_configure_hook(context_var, inheritable, handle_class=None, env_var=None):
    _configure_hooks.append((context_var, inheritable, handle_class, env_var))
```

`UsageMetadataCallbackHandler` 就利用这个机制实现上下文管理器模式：

```python
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    llm_1.invoke("Hello")
    llm_2.invoke("Hello")
    print(cb.usage_metadata)
    # {"gpt-4o": {"input_tokens": 15, "output_tokens": 42, ...}}
```

## 第二层：Tracer 系统

### _TracerCore 与 BaseTracer

Tracer 是 Callback 的特化——每个 Tracer 都是一个 `BaseCallbackHandler`，但额外维护结构化的执行树。继承层级：

```
_TracerCore (ABC)          # 追踪核心逻辑：Run 创建、树结构管理、dotted_order
    |
    +-- BaseTracer         # 同步追踪基类（mixin BaseCallbackHandler）
    |       |
    |       +-- LangChainTracer   # LangSmith 集成
    |
    +-- AsyncBaseTracer    # 异步追踪基类（mixin AsyncCallbackHandler）
```

`_TracerCore` 管理两个关键数据结构：

- **`run_map: dict[str, Run]`** -- 活跃 Run 的索引，通过 `run_id` 查找
- **`order_map: dict[UUID, tuple[UUID, str]]`** -- Run 的追踪排序信息 `(trace_id, dotted_order)`

`dotted_order` 是 LangSmith 的排序方案，格式为时间戳+ID的点分层级字符串，允许在分布式系统中按因果关系排序 Run：

```python
def _start_trace(self, run):
    current_dotted_order = run.start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id)
    if run.parent_run_id:
        parent = self.order_map.get(run.parent_run_id)
        run.trace_id, run.dotted_order = parent
        run.dotted_order += "." + current_dotted_order  # 追加到父级
    else:
        run.trace_id = run.id
        run.dotted_order = current_dotted_order  # 根节点
```

### LangChainTracer：LangSmith 的桥梁

`LangChainTracer` 是连接 LangChain 和 LangSmith 商业平台的核心。它的 `run_inline = True` 意味着在异步上下文中也会被串行执行，保证事件顺序。

每个 Run 在 `start` 时通过 `_persist_run_single` 发送 POST 请求到 LangSmith，在 `end/error` 时通过 `_update_run_single` 发送 PATCH 更新。Token 级事件经过去重处理——只记录首个 token 事件：

```python
def _llm_run_with_token_event(self, token, run_id, chunk=None, ...):
    run_id_str = str(run_id)
    if run_id_str not in self.run_has_token_event_map:
        self.run_has_token_event_map[run_id_str] = True
    else:
        return self._get_run(run_id, ...)  # 后续 token 不再记录事件
```

`LangSmithParams` TypedDict 为追踪提供模型元数据：

```python
class LangSmithParams(TypedDict, total=False):
    ls_provider: str          # "openai", "anthropic" 等
    ls_model_name: str        # "gpt-4o", "claude-3-opus" 等
    ls_model_type: Literal["chat", "llm"]
    ls_temperature: float | None
    ls_max_tokens: int | None
    ls_stop: list[str] | None
```

### EventStream：astream_events v2

`_AstreamEventsCallbackHandler` 是 `astream_events` API 的内部实现。它将 Callback 事件转化为结构化的 `StreamEvent` 字典，通过内存流（`_MemoryStream`）异步推送给消费者：

```python
# 事件格式
{
    "event": "on_chat_model_stream",     # 事件类型
    "run_id": "...",                      # 当前 Run ID
    "name": "ChatOpenAI",                # 组件名称
    "tags": ["my-tag"],                  # 标签
    "metadata": {},                      # 元数据
    "data": {"chunk": AIMessageChunk(...)},  # 事件数据
    "parent_ids": ["root-id", "chain-id"],   # 祖先 Run ID 链
}
```

`_RootEventFilter` 支持按名称、类型和标签过滤事件：

```python
async for event in chain.astream_events(
    input,
    version="v2",
    include_types=["chat_model"],   # 只看模型事件
    exclude_names=["RunnableSequence"],  # 排除管道包装器事件
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

### LogStream：JSON Patch 增量日志

`LogStreamCallbackHandler` 是另一种追踪模式——它将整个执行过程建模为一棵 `RunLog` 树，通过 JSON Patch（RFC 6902）增量更新：

```python
async for patch in chain.astream_log(input):
    # patch.ops 是 JSON Patch 操作列表
    # [{"op": "add", "path": "/logs/ChatOpenAI/streamed_output/-", "value": ...}]
    pass
```

`astream_log` 已被 `astream_events` 取代，但在需要完整 Run 树快照的场景下仍有价值。

## UsageMetadataCallbackHandler：Token 消耗统计

`UsageMetadataCallbackHandler` 展示了回调系统在实际场景中的用法——跨多次调用聚合 Token 消耗：

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
llm_1.invoke("Hello", config={"callbacks": [callback]})
llm_2.invoke("Hello", config={"callbacks": [callback]})

print(callback.usage_metadata)
# {
#     "gpt-4o": {"input_tokens": 15, "output_tokens": 42, "total_tokens": 57},
#     "claude-3-haiku": {"input_tokens": 12, "output_tokens": 38, "total_tokens": 50}
# }
```

它通过 `on_llm_end` 钩子从 `AIMessage.usage_metadata` 中提取统计信息，使用线程锁 (`threading.Lock`) 保证并发安全，按 `model_name` 聚合。

## StdOutCallbackHandler：最简调试

`StdOutCallbackHandler` 是最直接的调试工具。它只关注 Chain 的进出和 Agent 的动作：

```python
from langchain_core.callbacks import StdOutCallbackHandler

chain.invoke(input, config={"callbacks": [StdOutCallbackHandler()]})
# 输出：
# > Entering new RunnableSequence chain...
# > Finished chain.
```

设置 `langchain.debug = True` 会注入更详细的 `ConsoleCallbackHandler`（位于 `tracers/stdout.py`），它输出每个 Run 的完整输入输出，是排查 LCEL 管道问题的首选手段。

## 自定义事件：dispatch_custom_event

LangChain 还提供了用户自定义事件的能力，允许在 RunnableLambda 或 Tool 内部发送自定义事件：

```python
from langchain_core.callbacks import dispatch_custom_event

def my_step(inputs):
    # 处理中间结果时发送自定义事件
    dispatch_custom_event("progress", {"step": 1, "total": 3})
    # ...
    return result
```

这些事件会出现在 `astream_events` 流中，也会被注册了 `on_custom_event` 的 handler 接收。限制是必须在某个 Run 的上下文中调用——`dispatch_custom_event` 会检查 `callback_manager.parent_run_id` 是否存在。

## 设计思考

### 为什么选择回调而非 OpenTelemetry

LangChain 的可观测性系统是自研的回调体系，而非基于 OpenTelemetry（OTel）等行业标准。这个选择有其合理性，也付出了代价。

**选择回调的理由：**

1. **LLM 调用的语义丰富度**。OTel 的 Span 模型适合 request/response 范式，但 LLM 调用有独特的中间事件——流式 token、工具调用、重试——这些不容易映射到标准 Span 属性。LangChain 的回调可以为每种事件定义专属的参数签名（`on_llm_new_token` 的 `chunk` 参数、`on_tool_start` 的 `inputs` 参数）。

2. **控制流需求**。回调不仅仅是观测——`astream_events` 需要将回调事件转化为异步流供消费者消费，`run_inline` 控制执行顺序。这超出了传统 tracing 的观测范畴，进入了控制流领域。

3. **历史时机**。LangChain 在 2022 年底诞生时，OTel 的 LLM 语义约定（Gen AI Semantic Conventions）尚未成形。框架需要立即可用的解决方案。

**付出的代价：**

1. **生态隔离**。Datadog、Jaeger、Grafana 等主流 APM 工具无法直接消费 LangChain 的追踪数据，需要专门的适配层。
2. **厂商耦合**。`_configure` 函数中硬编码了 `LangChainTracer` 和 `LANGCHAIN_TRACING_V2` 环境变量的检测逻辑。设置一个环境变量就自动发送数据到 LangSmith——便利性和耦合性的一体两面。
3. **维护成本**。自研追踪系统意味着要自行解决分布式追踪、采样、导出等问题，而这些都是 OTel 已经解决的。

### 框架级追踪与商业平台的耦合

`langchain-core` 对 `langsmith` Python 包有直接依赖，`LangChainTracer` 在 `_configure` 函数中被自动注入。这意味着 LangChain 的核心库和 LangChain 公司的商业产品之间存在代码级耦合。

从 `_configure` 函数可以看到这个耦合有多深：

```python
from langsmith.run_helpers import get_tracing_context  # 直接导入 langsmith
from langchain_core.tracers.langchain import LangChainTracer

tracing_context = get_tracing_context()
run_tree = tracing_context["parent"]  # 从 langsmith 的上下文中获取父 Run
```

`register_configure_hook` 机制试图缓解这个问题——第三方可以通过注册自己的 `ContextVar` 钩子来注入自定义 tracer。但核心路径上对 LangSmith 的优先级是明确的。

这种设计在开源框架中并不罕见（Elasticsearch 之于 Kibana、Grafana 之于 Grafana Cloud），但它确实引出一个问题：当你选择 LangChain 作为编排层时，你是否也在隐性地选择了 LangSmith 作为可观测性平台？技术上不是必须的，但阻力最小的路径指向那里。

> **下一章**: [10 - 生产实践：安全、性能与框架取舍](10-production-practices.md)
