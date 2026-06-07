# 05 - Pregel 执行引擎

## 为什么叫 Pregel

Google Pregel 是 2010 年发布的大规模图计算框架，用于 PageRank 等算法。它提出了 **BSP（Bulk Synchronous Parallel）** 计算模型。

LangGraph 借用了这个模型来编排 Agent 的执行流程——虽然 Agent 不是大规模图计算，但 BSP 模型解决了 Agent 编排的核心问题：**如何在有状态的多步骤执行中保持一致性**。

## BSP 模型在 LangGraph 中的体现

原始 BSP 模型：
```
SuperStep N:  所有活跃 Processor 并行计算
              ↓
Barrier:      同步点 — 等所有 Processor 完成
              ↓
Communication: 交换消息
              ↓
SuperStep N+1: ...
```

LangGraph 的映射：
```
SuperStep N:  所有活跃 Node 并行执行
              ↓
Barrier:      等所有 Node 完成
              ↓
Channel 写入:  节点输出通过 Reducer 合并到 Channel
Checkpoint:    序列化所有 Channel 状态
Stream:        发出事件
              ↓
SuperStep N+1: 根据 Edge 确定下一批活跃 Node
```

## 三层执行架构

### Pregel — 总指挥

```python
# source/libs/langgraph/langgraph/pregel/main.py (~4335 LOC)
```

`Pregel` 是编译后的图的运行时表示。`CompiledStateGraph` 继承自 `Pregel`。

核心职责：
- **对外 API**：`invoke()`, `stream()`, `ainvoke()`, `astream()`
- **状态管理**：`get_state()`, `get_state_history()`, `update_state()`
- **生命周期**：初始化 channels → 加载 checkpoint → 运行 loop → 返回结果

```python
# 简化的 invoke 流程
def invoke(self, input, config):
    # 1. 获取或创建 checkpoint
    checkpoint = self.checkpointer.get(config)

    # 2. 从 checkpoint 恢复 channels
    channels = restore_channels(checkpoint)

    # 3. 将 input 写入 channels
    apply_input(channels, input)

    # 4. 创建 loop 并运行到完成
    loop = PregelLoop(channels, self.nodes, self.edges, ...)
    while loop.tick():
        pass

    # 5. 读取最终 channel 值作为输出
    return read_output(channels)
```

### PregelLoop — 节拍器

```python
# source/libs/langgraph/langgraph/pregel/_loop.py (~1963 LOC)
```

管理 SuperStep 循环，是执行流程的控制中枢。

两个核心方法：

**`tick()`** — 准备下一步
```python
def tick(self):
    # 1. 根据 edges + channel 状态确定哪些节点应该被激活
    tasks = prepare_tasks(self.nodes, self.edges, self.channels)

    # 2. 如果没有待执行的任务，返回 False（循环结束）
    if not tasks:
        return False

    # 3. 检查 interrupt_before：某些节点需要在执行前暂停
    if should_interrupt_before(tasks):
        raise GraphInterrupt(...)

    # 4. 交给 Runner 执行
    self.runner.tick(tasks)

    return True
```

**`after_tick()`** — 每步收尾
```python
def after_tick(self):
    # 1. 将节点输出写入 channels（经过 reducer）
    apply_writes(self.channels, task_results)

    # 2. 创建 checkpoint
    self.checkpointer.put(serialize(self.channels))

    # 3. 发出 stream 事件
    self.stream.emit(events)

    # 4. 检查 interrupt_after
    if should_interrupt_after(completed_nodes):
        raise GraphInterrupt(...)

    # 5. 递增 step 计数器，检查 recursion limit
    self.step += 1
    if self.step >= self.recursion_limit:
        raise GraphRecursionError(...)
```

### PregelRunner — 执行器

```python
# source/libs/langgraph/langgraph/pregel/_runner.py (~941 LOC)
```

在单个 SuperStep 内并行执行所有被激活的任务。

```python
def tick(self, tasks):
    # 并行执行所有 tasks
    futures = [executor.submit(task.proc, task.input) for task in tasks]

    # 等待所有完成（带重试逻辑）
    for future in as_completed(futures):
        try:
            result = future.result()
            task.writes = result
        except Exception as e:
            if should_retry(task, e):
                retry(task)
            else:
                raise
```

## 执行过程详解

以一个简单的 Agent 为例：

```python
graph = StateGraph(MessagesState)
graph.add_node("llm", call_llm)
graph.add_node("tools", run_tools)
graph.add_conditional_edges("llm", should_use_tools, {
    "yes": "tools",
    "no": END,
})
graph.add_edge("tools", "llm")
graph.add_edge(START, "llm")
```

执行流程：

```
输入: {"messages": [HumanMessage("天气怎么样？")]}

SuperStep 0 — 处理输入
  ├── input → channels["messages"].update([HumanMessage("天气怎么样？")])
  └── checkpoint #0

SuperStep 1 — 执行 "llm" 节点
  ├── 读: channels["messages"].get() → [HumanMessage("天气怎么样？")]
  ├── 执行: call_llm(state) → AIMessage(tool_calls=[get_weather(...)])
  ├── 写: channels["messages"].update([AIMessage(...)])
  ├── 路由: should_use_tools(state) → "yes" → 下一步激活 "tools"
  └── checkpoint #1

SuperStep 2 — 执行 "tools" 节点
  ├── 读: channels["messages"].get() → [Human, AI]
  ├── 执行: run_tools(state) → ToolMessage("北京 25°C 晴")
  ├── 写: channels["messages"].update([ToolMessage(...)])
  ├── Edge: tools → llm → 下一步激活 "llm"
  └── checkpoint #2

SuperStep 3 — 再次执行 "llm" 节点
  ├── 读: channels["messages"].get() → [Human, AI, Tool]
  ├── 执行: call_llm(state) → AIMessage("今天北京25°C，天气晴朗")
  ├── 写: channels["messages"].update([AIMessage(...)])
  ├── 路由: should_use_tools(state) → "no" → END
  └── checkpoint #3

完成: 无活跃节点 → 返回最终 state
```

## Recursion Limit

防止无限循环的保护机制：

```python
app = graph.compile()
app.invoke(input, config={"recursion_limit": 50})  # 最多 50 个 SuperStep
```

默认值 1000（从 v1.0.6 开始）。

### 主动 vs 被动处理

```python
from langgraph.managed import RemainingSteps

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining: RemainingSteps  # 可注入的剩余步数

def my_node(state):
    if state["remaining"] < 5:
        # 主动降级：还有不到 5 步时，直接给出最终答案
        return {"messages": [AIMessage("让我总结一下目前的发现...")]}
    else:
        # 正常处理
        return {"messages": [call_llm(state["messages"])]}
```

| 策略 | 触发时机 | 行为 |
|------|---------|------|
| 被动（默认） | 达到 limit | 抛出 `GraphRecursionError` |
| 主动（推荐） | 接近 limit | 节点自行降级、收尾 |

## Channel 读写机制

### ChannelRead — 读取

```python
# source/libs/langgraph/langgraph/pregel/_read.py
```

每个节点执行前，`ChannelRead` 从指定的 channels 中读取值，组装成 state dict 传给节点函数。

### ChannelWrite — 写入

```python
# source/libs/langgraph/langgraph/pregel/_write.py
```

节点返回的 dict 通过 `ChannelWrite` 分发到各个 channel，触发各自的 `update()` 方法（即 reducer）。

写入顺序和一致性保证：
- 同一个 SuperStep 内的所有写入在**该步结束后**才可见
- 下一个 SuperStep 的节点才能读到本步的写入结果
- 这保证了**步内一致性**——一个节点不会看到同步执行的另一个节点的中间结果

## 重试机制

```python
# source/libs/langgraph/langgraph/pregel/_retry.py (~854 LOC)
```

节点执行失败时的重试策略：

```python
from langgraph.pregel import RetryPolicy

graph.add_node("llm", call_llm, retry=RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,   # 首次重试等待 1 秒
    backoff_factor=2.0,     # 指数退避
    max_interval=10.0,      # 最长等待 10 秒
))
```

重试发生在 `PregelRunner` 层面——单个任务失败会重试，不影响同步执行的其他任务。

## 设计思考

### 为什么选 BSP 而不是 Actor Model？

Actor Model（如 Erlang/Akka）是另一种并发编排方案。对比：

| 维度 | BSP (LangGraph) | Actor Model |
|------|-----------------|-------------|
| 同步模型 | 全局同步（SuperStep 屏障） | 完全异步 |
| 状态一致性 | 每步结束后全局一致 | 最终一致 |
| 调试难度 | 低——状态变化是离散的步 | 高——消息顺序不确定 |
| 持久化 | 天然适合——每步一个快照 | 困难——没有自然的快照点 |
| 性能 | 受限于最慢的节点（barrier） | 更高（无等待） |

对于 Agent 场景，BSP 的**可预测性和可调试性**比 Actor 的性能优势更重要。LLM 调用本身就是耗时操作（秒级），barrier 的开销可以忽略。

### 为什么 Pregel 是最大的模块？

`pregel/` 目录约 12,000 行代码，远超其他模块，因为它要处理：

1. 同步和异步两套执行路径
2. 流式输出的事件发射
3. Checkpoint 的保存和恢复
4. Interrupt 的抛出和恢复
5. 重试和错误处理
6. 子图的递归执行
7. 远程图（LangGraph Server）的代理

这些关注点交织在一起，导致了复杂度集中。

> **下一章**: [06 - 持久化与 Checkpointer](06-checkpointer.md) — 理解 LangGraph 的状态持久化机制
