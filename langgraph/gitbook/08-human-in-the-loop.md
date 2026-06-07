# 08 - Human-in-the-Loop

## 为什么 Agent 需要人工介入

自主 Agent 不可能 100% 可靠。在以下场景中，人工介入是必须的：

| 场景 | 例子 |
|------|------|
| **高风险操作** | 删除数据、发送邮件、执行交易 |
| **不确定决策** | Agent 不确定该走哪条路径 |
| **信息补充** | 需要用户提供额外信息 |
| **质量审核** | 在提交前审核 Agent 的输出 |

LangGraph 的 HITL 机制基于 Checkpointer 实现——**暂停 = 创建 checkpoint + 抛出异常，恢复 = 从 checkpoint 加载 + 继续执行**。

## 三种中断方式

### 方式 1：`interrupt()` 函数（最灵活）

```python
from langgraph.types import interrupt

def review_node(state):
    # 收集需要审核的内容
    draft = state["draft"]

    # 暂停，等待人工决策
    decision = interrupt({
        "question": "是否批准这份草稿？",
        "draft": draft,
    })

    # 恢复后，decision 包含人工的回复
    if decision["approved"]:
        return {"status": "approved"}
    else:
        return {"status": "rejected", "feedback": decision["reason"]}
```

`interrupt()` 做了什么：
1. 序列化参数作为中断信息
2. 抛出 `GraphInterrupt` 异常
3. Pregel 引擎捕获异常，创建 checkpoint
4. 执行暂停，控制权返回给调用者

### 方式 2：`interrupt_before` / `interrupt_after`（编译时配置）

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_node"],   # 执行前暂停
    interrupt_after=["review_node"],       # 执行后暂停
)
```

这种方式不需要修改节点代码——在**图的编译级别**指定哪些节点需要暂停。

区别：
- `interrupt_before` — 节点还没执行，状态是执行前的状态。适合"确认是否要执行"
- `interrupt_after` — 节点已执行，状态包含执行结果。适合"审核执行结果"

### 方式 3：结合 `Command` 恢复

```python
from langgraph.types import Command

# 恢复执行（传入 interrupt 的回复）
result = app.invoke(
    Command(resume={"approved": True}),
    config,
)

# 恢复执行 + 修改状态 + 跳转到特定节点
result = app.invoke(
    Command(
        resume={"approved": False},
        update={"status": "rejected"},
        goto="error_handler",
    ),
    config,
)
```

## 完整的 HITL 流程

```
第一次调用
┌─────────────────────────────────────────┐
│ app.invoke(input, config)               │
│   ├── START → llm → tools → review     │
│   │                          ↓          │
│   │                    interrupt()      │
│   │                          ↓          │
│   └── 返回（带 interrupts 信息）         │
└─────────────────────────────────────────┘
         │
         ▼  人工审核（可能是 UI、API、Slack...）
         │
┌─────────────────────────────────────────┐
│ app.invoke(Command(resume=answer), cfg) │
│   ├── review 恢复执行                    │
│   │   ├── decision = answer             │
│   │   └── 返回 {"status": "approved"}   │
│   ├── → finalize → END                  │
│   └── 返回最终结果                       │
└─────────────────────────────────────────┘
```

## 状态修改

除了简单地恢复，人工还可以在恢复前**修改** Agent 的状态：

### 修改消息

```python
# 读取当前状态
state = graph.get_state(config)

# 修改 LLM 的输出（比如纠正错误的 tool call）
from langchain_core.messages import AIMessage
corrected_msg = AIMessage(
    content="",
    tool_calls=[{"name": "correct_tool", "args": {...}, "id": "call_1"}],
    id=state.values["messages"][-1].id,  # 用相同 ID 替换
)

# 更新状态（创建新 checkpoint）
graph.update_state(config, {"messages": [corrected_msg]}, as_node="llm")

# 从修改后的状态继续
result = graph.invoke(None, config)
```

### `as_node` 的作用

`update_state(config, values, as_node="X")` 让图认为这个更新是节点 X 产生的。这决定了恢复后走哪条 edge：

```python
# 假装是 "llm" 节点产生的 → 走 llm 的 conditional edge
graph.update_state(config, values, as_node="llm")

# 假装是 "tools" 节点产生的 → 走 tools 的出边（回到 llm）
graph.update_state(config, values, as_node="tools")
```

## 多次中断

一个节点可以有多个 `interrupt()` 调用——每次恢复时执行到下一个 `interrupt`：

```python
def multi_step_review(state):
    # 第一次中断：审核内容
    content_decision = interrupt({"step": "review_content", "content": state["draft"]})
    if not content_decision["ok"]:
        return {"status": "rejected"}

    # 第二次中断：确认发送
    send_decision = interrupt({"step": "confirm_send", "recipients": state["to"]})
    if not send_decision["ok"]:
        return {"status": "cancelled"}

    send_email(state)
    return {"status": "sent"}
```

## 设计思考

### HITL 为什么依赖 Checkpointer？

因为"暂停"本质上就是"保存当前状态后退出"。如果没有 Checkpointer：
- 进程退出 → 状态丢失
- 无法跨请求恢复
- 无法在分布式环境中恢复（不同 server 处理恢复请求）

所以 **HITL 必须配合 Checkpointer 使用**，`interrupt()` 在没有 Checkpointer 时会报错。

### Interrupt vs Callback 的设计选择

很多框架用 callback 实现人工介入：

```python
# callback 模式（其他框架）
def node(state, on_approval):
    result = do_something()
    approved = on_approval(result)  # 阻塞等待
    ...
```

LangGraph 选择了 interrupt 模式，原因：
1. **不阻塞进程**——interrupt 后进程释放，可以服务其他请求
2. **天然支持异步审核**——可以等几分钟、几小时甚至几天
3. **可以跨进程恢复**——不同的 server 可以处理恢复请求
4. **审核者可以修改状态**——不只是 yes/no，还能改 Agent 的决策

代价是：API 更复杂（需要两次调用），开发者需要理解 checkpoint 的概念。

> **下一章**: [09 - 工作流模式](09-workflow-patterns.md) — 6 种常见的 Agent 编排模式
