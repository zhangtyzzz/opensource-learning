# 04 - Channel 机制

## Channel 是什么

Channel 是 LangGraph 中**最底层的状态管理原语**。当你定义一个 State：

```python
class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
```

编译时，LangGraph 会为每个字段创建一个 Channel：
- `messages` → `BinaryOperatorAggregate` channel（因为有 `add_messages` reducer）
- `name` → `LastValue` channel（没有 reducer，默认行为）

**你永远不需要直接操作 Channel**——它是 State 声明和 Pregel 引擎之间的内部抽象。但理解它有助于理解很多"为什么"的问题。

## Channel 的生命周期

```
from_checkpoint()  ← 从 checkpoint 恢复状态
    │
    ▼
get()  ← 节点读取当前值
    │
    ▼
update(writes)  ← 节点写入后，通过 reducer 合并
    │
    ▼
checkpoint()  ← 序列化当前状态用于持久化
    │
    ▼
consume()  ← 某些 channel 会在消费后重置（如 EphemeralValue）
```

对应源码中的 `BaseChannel` 抽象类：

```python
# source/libs/langgraph/langgraph/channels/base.py
class BaseChannel(ABC):
    def from_checkpoint(self, checkpoint)   # 从持久化恢复
    def get(self)                           # 读取当前值
    def update(self, values)                # 接收写入
    def checkpoint(self)                    # 导出用于持久化
    def consume(self)                       # 消费后重置（可选）
    def finish(self)                        # 一轮结束后的清理
```

## Channel 类型详解

### LastValue — 最后写入者胜

```python
# source/libs/langgraph/langgraph/channels/last_value.py
```

行为：保存最新写入的值，新值覆盖旧值。

```python
class MyState(TypedDict):
    name: str           # → LastValue channel
    current_step: str   # → LastValue channel
```

规则：
- 同一个 SuperStep 内只允许一个节点写入（否则报错）
- 这意味着 **LastValue 字段天然不支持并行写入**

### BinaryOperatorAggregate — Reducer 聚合

```python
# source/libs/langgraph/langgraph/channels/binop.py
```

行为：通过一个二元操作符（reducer）合并多次写入。

```python
class MyState(TypedDict):
    # operator.add → 列表拼接
    messages: Annotated[list, operator.add]

    # 自定义 reducer → 任意合并逻辑
    count: Annotated[int, lambda old, new: old + new]
```

规则：
- 同一个 SuperStep 内多个节点可以并行写入（reducer 会合并）
- Reducer 必须是**结合律**的：`f(f(a, b), c) == f(a, f(b, c))`

### Topic — 发布/订阅

```python
# source/libs/langgraph/langgraph/channels/topic.py
```

行为：多值通道，一个 SuperStep 内可以有多个值，消费后清空。

与 `BinaryOperatorAggregate` 的区别：
- Topic 不聚合——每个值独立存在
- Topic 每步消费后清空——下一步读到的只有当步的新写入

适用场景：事件广播、消息队列。

### EphemeralValue — 单步临时值

```python
# source/libs/langgraph/langgraph/channels/ephemeral_value.py
```

行为：值只存活一个 SuperStep，下一步自动清空。

```
SuperStep 1: 节点 A 写入 ephemeral_field = "hello"
SuperStep 2: 节点 B 读 ephemeral_field → EmptyChannelError（已清空）
```

适用场景：只需要传递给直接下游的临时数据。

### DeltaChannel — 增量存储（Beta）

```python
# source/libs/langgraph/langgraph/channels/delta.py
```

行为：只持久化增量（delta），不是完整状态。

```python
class MyState(TypedDict):
    messages: Annotated[list[str], DeltaChannel(
        reducer=append,
        snapshot_frequency=50  # 每 50 步做一次全量快照
    )]
```

恢复时：找最近的全量快照 + 回放后续增量。

这是 LangGraph 对 [Checkpointer O(N^2) 问题](../notes/01-checkpointer-deep-dive.md) 的官方解决方案。

### NamedBarrierValue — 同步屏障

```python
# source/libs/langgraph/langgraph/channels/named_barrier_value.py
```

行为：等待所有指定的"参与者"都写入后才释放。

内部使用，用于 `START` → 多节点 → 汇聚节点 的同步控制。你不需要直接使用它。

## Channel 与 Checkpoint 的交互

每个 SuperStep 结束后，所有 Channel 的状态被序列化为一个 checkpoint：

```python
# 简化的 checkpoint 结构
{
    "channel_values": {
        "messages": serialize(channel.checkpoint()),  # → checkpoint_blobs 表
        "name": serialize(channel.checkpoint()),
    },
    "channel_versions": {
        "messages": "v3",  # 版本号，用于增量恢复
        "name": "v1",
    }
}
```

关键优化：如果一个 Channel 在某个 SuperStep 中没有被写入，它的版本号不变，Checkpointer 可以复用之前的 blob——这就是 `checkpoint_blobs` 表的 `(channel, version)` 复合主键的意义。

## 设计思考

### 为什么需要这么多 Channel 类型？

核心矛盾：**不同的数据有不同的更新语义**。

- 用户名 → 覆盖（LastValue）
- 消息列表 → 追加（BinaryOperatorAggregate）
- 临时搜索结果 → 用完即弃（EphemeralValue）
- 事件通知 → 广播后清空（Topic）

如果强行用一种 Channel 类型，要么丢失语义（全用覆盖），要么过度复杂（全用 reducer）。

### Channel 类型选择指南

```
这个字段需要合并多次写入吗？
  ├── 不需要 → LastValue（默认）
  └── 需要
      ├── 合并后保留全部 → BinaryOperatorAggregate（用 operator.add）
      ├── 合并后只保留一个 → BinaryOperatorAggregate（自定义 reducer）
      └── 每步独立，不合并 → Topic

这个字段的值需要跨 SuperStep 保留吗？
  ├── 需要 → 以上任一
  └── 不需要 → EphemeralValue
```

> **下一章**: [05 - Pregel 引擎](05-pregel-engine.md) — LangGraph 最核心的执行引擎
