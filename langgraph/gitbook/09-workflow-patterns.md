# 09 - 工作流模式

## Workflow vs Agent

```
Workflow: 预定义的执行路径，人决定流程
Agent:    动态的执行路径，LLM 决定流程
```

LangGraph 同时支持两种模式，而且它们可以自由组合——一个 Workflow 中的某个节点可以是一个 Agent。

## 6 种编排模式

### 模式 1：Prompt 链（Sequential）

```
A → B → C → END
```

场景：每步依赖上一步的输出。例如：生成大纲 → 写内容 → 润色。

```python
graph.add_edge(START, "generate_outline")
graph.add_edge("generate_outline", "write_content")
graph.add_edge("write_content", "polish")
graph.add_edge("polish", END)
```

可以加**条件门**（quality check）在任意步之间短路。

### 模式 2：并行化（Fan-out / Fan-in）

```
START → ┌ A ┐
        │ B │ → Aggregate → END
        └ C ┘
```

场景：多个独立任务并行处理后汇总。例如：同时搜索多个来源 → 合并结果。

```python
graph.add_edge(START, "search_web")
graph.add_edge(START, "search_db")
graph.add_edge(START, "search_docs")
graph.add_edge("search_web", "aggregate")
graph.add_edge("search_db", "aggregate")
graph.add_edge("search_docs", "aggregate")
```

汇聚节点的 State 字段需要用 Reducer（如 `operator.add`）来合并多个并行节点的输出。

### 模式 3：路由（Router）

```
START → Router → ┌ Handler A ┐
                  │ Handler B │ → END
                  └ Handler C ┘
```

场景：根据输入分类到不同处理器。例如：意图识别后分流。

```python
graph.add_conditional_edges("router", classify_intent, {
    "billing": "billing_handler",
    "technical": "tech_handler",
    "general": "general_handler",
})
```

路由函数通常用 `with_structured_output` 获得类型安全的分类结果。

### 模式 4：Orchestrator-Worker（动态分工）

```
Orchestrator → ┌ Worker 1 ┐
               │ Worker 2 │ → Synthesizer → END
               │ ...      │
               └ Worker N ┘
```

场景：任务数量在运行时才确定。例如：分析文章 → 拆成 N 个章节 → 每个章节独立处理。

关键 API — **`Send`**：

```python
from langgraph.types import Send

def orchestrator(state):
    plan = llm.invoke("请将以下任务拆分为子任务: " + state["task"])
    # 动态创建 N 个 worker
    return [Send("worker", {"subtask": t}) for t in plan.subtasks]

graph.add_conditional_edges("orchestrator", orchestrator)
```

`Send` 和静态并行（多条 edge）的区别：
- 静态并行：编译时就知道有几个分支
- `Send`：运行时决定分支数量和输入

### 模式 5：Evaluator-Optimizer（迭代优化）

```
Generator ←──── Evaluator
    │               │
    │    不合格      │ 合格
    │               ▼
    └──────────   END
```

场景：生成 → 评估 → 不满意就带反馈重新生成。例如：写代码 → 跑测试 → 失败就修复。

```python
graph.add_conditional_edges("evaluator", check_quality, {
    "pass": END,
    "fail": "generator",  # 带反馈重新生成
})
```

注意设置 `recursion_limit` 防止无限循环。

### 模式 6：ReAct Agent（工具循环）

```
LLM ←──── Tools
 │           │
 │  有工具调用 │
 │           │
 │  无工具调用
 ▼
END
```

场景：最经典的 Agent 模式——LLM 决定是否调用工具，调用结果反馈给 LLM。

```python
graph.add_conditional_edges("llm", should_call_tools, {
    "yes": "tools",
    "no": END,
})
graph.add_edge("tools", "llm")
```

LangGraph 提供了预构建的实现：

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools, checkpointer=checkpointer)
```

## 模式选型指南

```
任务是否可以预定义步骤？
├── 是 → 步骤间是否有依赖？
│       ├── 有依赖 → Prompt 链
│       └── 无依赖 → 并行化
├── 需要分类 → 路由
├── 需要动态分解 → Orchestrator-Worker
├── 需要迭代改进 → Evaluator-Optimizer
└── 需要 LLM 自主决策 → ReAct Agent
```

## 模式组合

实际生产中，通常是多种模式的组合：

```
路由 → ┌ 简单查询: Prompt 链
       │ 复杂查询: ReAct Agent
       └ 批量任务: Orchestrator-Worker
                      └ 每个 Worker: Evaluator-Optimizer
```

在 LangGraph 中，组合通过**子图（Subgraph）**实现——每种模式编译为独立的图，然后作为节点嵌入到上层图中。

> **下一章**: [10 - 生产实践](10-production.md) — 部署、性能、常见陷阱
