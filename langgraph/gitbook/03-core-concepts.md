# 03 - 核心概念

## 概念关系全景

```
StateGraph ──compile()──▶ CompiledStateGraph (Pregel)
    │                           │
    │ 定义                      │ 运行时
    │                           │
    ├── State (TypedDict)       ├── Channels (一个字段一个 Channel)
    │     └── Reducers          │     ├── LastValue
    │                           │     ├── BinaryOperatorAggregate
    │                           │     └── ...
    ├── Nodes (函数)            │
    │                           ├── SuperStep 循环
    └── Edges (连接关系)        │
          ├── Normal Edge       └── Checkpoint (每步快照)
          ├── Conditional Edge
          └── Entry/Finish Points
```

## State

State 是图中所有节点共享的数据容器。它定义了"这个图在处理什么数据"。

### 三种 Schema 类型

```python
# 方式 1: TypedDict（推荐，最常用）
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_step: str

# 方式 2: dataclass（需要默认值时）
@dataclass
class AgentState:
    messages: Annotated[list, add_messages] = field(default_factory=list)
    current_step: str = "init"

# 方式 3: Pydantic BaseModel（需要运行时校验时，有性能开销）
class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    current_step: str = "init"
```

### Reducer：字段如何被更新

每个 State 字段可以有自己的 Reducer，决定"多个节点写同一个字段时怎么合并"。

```python
class MyState(TypedDict):
    # 没有 Reducer → 后写覆盖前写（LastValue Channel）
    name: str

    # 有 Reducer → 通过 operator.add 合并（BinaryOperatorAggregate Channel）
    messages: Annotated[list, operator.add]

    # 用 add_messages → 智能合并（按 message ID 去重）
    chat: Annotated[list, add_messages]
```

**Reducer 的本质**：它决定了这个字段对应哪种 Channel 类型。这就是 State 定义和底层 Channel 系统的桥梁。

### 重要细节：节点可以写任意字段

一个经常被忽略的事实：**节点可以写 State 中的任何字段**，不受其输入 schema 限制。

```python
# 节点只接收 messages，但可以写 current_step
def my_node(state: AgentState) -> dict:
    return {
        "messages": [new_message],
        "current_step": "done"  # 可以写任何 State 字段
    }
```

### 多 Schema 模式

LangGraph 支持为图定义不同的输入/输出 Schema：

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class FullState(InputState, OutputState):
    intermediate_steps: list  # 内部字段，不暴露

graph = StateGraph(FullState, input=InputState, output=OutputState)
```

这样调用者只看到 `question` → `answer`，内部处理细节被隐藏。

## Node

Node 就是一个 Python 函数，接收 State，返回 State 的部分更新。

### 函数签名

```python
# 最简形式：只接收 state
def simple_node(state: MyState) -> dict:
    return {"field": "value"}

# 完整形式：state + config + runtime
def full_node(state: MyState, config: RunnableConfig, runtime: Runtime) -> dict:
    # config: thread_id, tags, metadata 等配置
    # runtime: store, stream_writer, execution_info 等运行时能力
    return {"field": "value"}
```

### 幂等性要求

**关键认知**：Checkpoint 保存在 SuperStep 边界，不是在函数执行中间。如果一个节点执行到一半被中断（比如 `interrupt()`），恢复后它会**从头重新执行**。

```python
# ❌ 不幂等：重执行会发两次邮件
def bad_node(state):
    send_email(state["draft"])  # 副作用！
    return interrupt("approve?")

# ✅ 幂等：用 Task 包装，已完成的 task 会跳过
@task
def send_email_task(draft):
    send_email(draft)

def good_node(state):
    send_email_task(state["draft"]).result()
    return interrupt("approve?")
```

### Task：更细粒度的检查点

`@task` 装饰器把一个操作包装为可单独检查点的单元。在节点被中断后重新执行时，已完成的 task 会跳过：

```python
from langgraph.func import task

@task
def fetch_data(url: str) -> dict:
    return requests.get(url).json()  # 耗时操作

def my_node(state):
    # 如果节点被重新执行，已完成的 task 直接返回缓存结果
    result1 = fetch_data("url1").result()
    result2 = fetch_data("url2").result()
    return {"data": [result1, result2]}
```

## Edge

Edge 定义了节点间的执行顺序。

### 四种 Edge 类型

```python
graph = StateGraph(MyState)

# 1. Normal Edge：A 执行完一定执行 B
graph.add_edge("A", "B")

# 2. Conditional Edge：根据函数返回值决定走向
graph.add_conditional_edges("A", route_function, {
    "option1": "B",
    "option2": "C",
})

# 3. Entry Point：图的起点
graph.add_edge(START, "A")

# 4. Finish Point：图的终点
graph.add_edge("B", END)
```

### 并行执行

**多条出边 = 并行**。如果一个节点有多条出边指向不同节点，这些目标节点会在下一个 SuperStep 中并行执行：

```python
graph.add_edge(START, "fetch_news")
graph.add_edge(START, "fetch_weather")
graph.add_edge(START, "fetch_stocks")
# → 三个节点同时执行
```

### Send：动态并行

当并行的数量在编译时未知（比如"处理搜索返回的每一条结果"），用 `Send`：

```python
from langgraph.types import Send

def router(state):
    # 运行时决定创建多少个并行分支
    return [Send("process_item", {"item": item}) for item in state["items"]]

graph.add_conditional_edges("search", router)
```

### Command：从节点内部控制流程

`Command` 允许节点在返回状态更新的同时，指定下一步跳转：

```python
from langgraph.types import Command

def smart_node(state):
    if state["score"] > 0.9:
        return Command(update={"status": "approved"}, goto="finalize")
    else:
        return Command(update={"status": "needs_review"}, goto="review")
```

**注意**：不要同时对一个节点设置 normal edge 和 Command——它们会冲突。

## SuperStep：执行的节拍

SuperStep 是 LangGraph 执行模型的核心节奏：

```
SuperStep 0: 处理输入 → 写入 channels
    ↓ checkpoint
SuperStep 1: 激活起始节点 → 并行执行 → 写入 channels
    ↓ checkpoint
SuperStep 2: 根据 edges 激活下一批节点 → 并行执行 → 写入 channels
    ↓ checkpoint
...直到没有活跃节点
```

每个 SuperStep 结束后：
1. 所有 channel 写入完成
2. 创建一个 checkpoint（状态快照）
3. 发出 stream 事件
4. 检查是否有 interrupt

这就是为什么 checkpoint 的粒度是 SuperStep 而不是单个节点——一个 SuperStep 可能包含多个并行节点的执行。

## 关键设计权衡

### 为什么选择 "State + Channel" 而不是 "消息传递"？

纯 Pregel 模型中，节点间通过显式消息通信。LangGraph 选择了共享状态 + Channel 的模式，原因是：

1. **降低心智负担**：Agent 开发者更习惯"读写共享状态"而非"发送/接收消息"
2. **更容易持久化**：状态是一个完整的 dict，序列化/反序列化直观
3. **更容易调试**：在任何时刻都能查看完整状态

代价是：**失去了纯消息传递的解耦性**。所有节点都能看到所有状态，没有天然的信息隔离。

### 为什么每个 SuperStep 都要 Checkpoint？

频繁 checkpoint 的好处：
- 精确的故障恢复（最多丢失一个 SuperStep 的工作）
- 支持 Human-in-the-Loop（任何 SuperStep 边界都可以暂停）
- 时间旅行调试

代价就是我们在 [notes/01-checkpointer-deep-dive.md](../notes/01-checkpointer-deep-dive.md) 中分析过的 O(N^2) 存储膨胀问题。

> **下一章**: [04 - Channel 机制](04-channels.md) — 理解 LangGraph 状态通信的底层原语
