# 06 - 持久化与 Checkpointer

## 为什么需要持久化

没有持久化的 Agent 是"一次性"的——崩溃就丢失进度，无法暂停恢复。LangGraph 的 5 个核心能力中有 4 个依赖持久化：

| 能力 | 对持久化的依赖 |
|------|--------------|
| Human-in-the-Loop | 必须。暂停 → 保存状态 → 恢复 |
| 记忆（Memory） | 必须。跨对话保持上下文 |
| 时间旅行（Time Travel） | 必须。回溯到任意历史快照 |
| 容错（Fault Tolerance） | 必须。从最近的 checkpoint 恢复 |
| 流式输出 | 不依赖 |

## 三个持久化抽象

```
Checkpointer — 线程内的状态快照序列
  "这个对话进行到哪一步了？"

Store — 跨线程的 KV 存储
  "这个用户喜欢什么？上次聊了什么？"

Cache — 节点执行结果缓存
  "相同输入不用重复计算"
```

### Checkpointer vs Store 的本质区别

```
                Thread A        Thread B        Thread C
              ┌──────────┐   ┌──────────┐   ┌──────────┐
Checkpointer  │ CP1→CP2→ │   │ CP1→CP2  │   │ CP1      │
(线程内)      │ CP3→CP4  │   │          │   │          │
              └──────────┘   └──────────┘   └──────────┘

                    ↕               ↕              ↕

Store          ┌─────────────────────────────────────────┐
(跨线程)       │  user:123 → {preferences: {...}}        │
               │  user:123:memories → [{...}, {...}]     │
               └─────────────────────────────────────────┘
```

## Checkpoint 机制

### 什么时候创建 Checkpoint

每个 **SuperStep 结束后**创建一个 checkpoint。对于 `START → A → B → END`，一次调用产生 **4 个 checkpoint**：

```
CP0: 初始空状态
CP1: 处理输入后
CP2: 节点 A 执行后
CP3: 节点 B 执行后
```

### StateSnapshot 结构

```python
snapshot = graph.get_state(config)

snapshot.values      # 当前 state dict
snapshot.next        # 下一步要执行的节点，() 表示已完成
snapshot.config      # {"thread_id": "...", "checkpoint_id": "..."}
snapshot.metadata    # {"source": "loop", "step": 3, "writes": {...}}
snapshot.created_at  # 时间戳
snapshot.parent_config  # 上一个 checkpoint 的 config
snapshot.tasks       # 当前步的任务信息
```

### 状态操作

**读取状态**
```python
# 最新状态
state = graph.get_state({"configurable": {"thread_id": "my-thread"}})

# 历史状态列表（最新在前）
for snapshot in graph.get_state_history(config):
    print(f"Step {snapshot.metadata['step']}: {snapshot.next}")
```

**修改状态（创建新的 checkpoint）**
```python
# 注意：这不是"修改"，而是"基于当前状态创建一个新 checkpoint"
graph.update_state(config, {"messages": [HumanMessage("新消息")]})

# as_node 参数：假装是某个节点产生的更新（影响 edge 路由）
graph.update_state(config, values, as_node="llm")
```

### 时间旅行

```python
# 找到特定步的 checkpoint
for snapshot in graph.get_state_history(config):
    if snapshot.metadata["step"] == 2:
        old_config = snapshot.config
        break

# 从那个点恢复执行
result = graph.invoke(None, old_config)
```

## Store：跨线程记忆

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
app = graph.compile(checkpointer=checkpointer, store=store)

# 在节点中访问 store
def my_node(state, config, runtime):
    user_id = config["configurable"]["user_id"]

    # 读取
    memories = runtime.store.search(("users", user_id, "memories"))

    # 写入
    runtime.store.put(("users", user_id, "memories"), "mem-1", {
        "content": "用户喜欢简洁的回答",
        "importance": 0.9,
    })

    return {"messages": [...]}
```

### 语义搜索

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["content"],  # 对哪些字段建向量索引
    }
)

# 语义搜索
results = store.search(
    ("users", user_id, "memories"),
    query="用户的偏好",
    limit=5,
)
```

## Checkpointer 实现

| 实现 | 适用场景 | 持久化 |
|------|---------|--------|
| `InMemorySaver` | 开发调试 | 进程内存，重启丢失 |
| `SqliteSaver` | 本地/小规模 | 本地文件 |
| `PostgresSaver` | 生产环境 | PostgreSQL |

### 持久化模式

```python
app = graph.compile(
    checkpointer=checkpointer,
    checkpoint_durability="exit",   # 默认：只在退出时持久化
    # "async"  — 异步持久化（性能与可靠性平衡）
    # "sync"   — 同步持久化（最可靠，最慢）
)
```

### 序列化

默认使用 `JsonPlusSerializer`（基于 orjson），支持 LangChain 的所有类型。

```python
# 加密序列化（生产推荐）
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

serializer = EncryptedSerializer.from_pycryptodome_aes()
# 需要设置 LANGGRAPH_AES_KEY 环境变量
```

## 存储膨胀问题

Checkpointer 的核心痛点是存储随对话增长而膨胀。这个问题在 [notes/01-checkpointer-deep-dive.md](../notes/01-checkpointer-deep-dive.md) 中有完整的深度分析，包括：

- 实际的 4 张表结构
- O(N^2) 增长的根本原因
- 6 种解决方案的对比
- DeltaChannel 的工作原理

**一句话总结**：存储优化（DeltaChannel）和内容裁剪（SummarizationMiddleware）是正交的两个问题，生产环境需要同时处理。

> **下一章**: [07 - Streaming 机制](07-streaming.md) — 7 种流模式详解
