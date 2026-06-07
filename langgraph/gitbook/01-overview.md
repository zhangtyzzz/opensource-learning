# 01 - 框架概览

## LangGraph 是什么

LangGraph 是一个**底层编排框架**，用于构建有状态的、长时间运行的 Agent 应用。

关键定位：它**不是** prompt 框架，不抽象 prompt 或模型调用方式。它解决的是更底层的问题——当你的 Agent 需要：

- 在多步骤之间维护状态
- 暂停等待人工审核后恢复
- 在崩溃后从断点继续
- 支持多个 Actor 并发执行

这些都是**基础设施问题**，而非 AI 问题。LangGraph 就是解决这类问题的运行时。

## 在 LangChain 生态中的位置

```
Deep Agents (应用层 Agent)
    ↓ 使用
LangChain (Agent 框架，提供 create_react_agent 等高层 API)
    ↓ 基于
LangGraph (编排运行时，提供 Graph/State/Checkpoint 等底层能力)
    ↓ 可选集成
LangSmith (可观测性、调试、部署)
```

你可以不用 LangChain，直接用 LangGraph 来编排任何 Python 代码——它不绑定 LangChain 的 Chat Model 或 Tool 抽象。

## 设计灵感

LangGraph 的三个关键灵感来源：

| 灵感来源 | 借鉴了什么 |
|---------|-----------|
| **Google Pregel** | BSP（Bulk Synchronous Parallel）计算模型：节点并行计算、通过消息通信、按 SuperStep 同步 |
| **Apache Beam** | 数据流编排的概念——把计算表达为 DAG |
| **NetworkX** | 图的构建 API 风格——`add_node`、`add_edge` |

其中 **Pregel 模型**是最核心的灵感。LangGraph 的执行引擎（`pregel/` 模块）就直接以 Pregel 命名。这意味着：

1. 计算被建模为**图中节点间的消息传递**
2. 节点默认是**不活跃**的，收到消息才激活
3. 一个 **SuperStep** 是所有活跃节点并行执行一轮
4. 执行终止条件：所有节点不活跃 + 没有消息在传输中

## 五个核心能力

| 能力 | 解决什么问题 |
|------|-------------|
| **持久化 (Persistence)** | Agent 可以在崩溃、超时、部署更新后从上次断点恢复 |
| **Human-in-the-Loop** | 在关键决策点暂停，等待人工审批或修改后继续 |
| **记忆 (Memory)** | 短期（对话内）和长期（跨对话）记忆支持 |
| **流式输出 (Streaming)** | 7 种流模式，支持 token 级别的实时输出 |
| **调试 (Debugging)** | 状态快照、时间旅行回放、与 LangSmith 集成 |

## 两种构建 API

LangGraph 提供两种风格的 API 来定义图，但它们**编译后产物完全相同**——都是 `Pregel` 实例：

### Graph API（声明式）

```python
from langgraph.graph import StateGraph, MessagesState, START, END

graph = StateGraph(MessagesState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
app = graph.compile()
```

适合：结构清晰的工作流，需要可视化图结构的场景。

### Functional API（命令式）

```python
from langgraph.func import entrypoint, task

@task
def chat(messages):
    return llm.invoke(messages)

@entrypoint()
def app(messages):
    return chat(messages).result()
```

适合：逻辑本身就是一个函数、不需要显式建图的场景。

### 选择建议

| 场景 | 推荐 |
|------|-----|
| 有明确的状态机 / 流程图 | Graph API |
| 需要可视化调试 | Graph API |
| 逻辑简单直接 | Functional API |
| 动态并行（数量不确定） | Functional API 更自然 |
| 两者混用 | 完全可以，`@task` 可以在 Graph 节点内使用 |

> **下一章**: [02 - 架构总览](02-architecture.md) — 深入 LangGraph 的分层架构和代码组织
