# LangGraph Checkpointer 机制深度分析

> 核心问题：随着对话越来越长，Checkpointer 存储的数据量会不会爆炸？
> 答案：会。LangChain 官方承认这是 O(N^2) 增长问题，并在 langgraph 1.2 中推出 DeltaChannel 作为解决方案。

## 1. 实际存储结构

Checkpointer **不是**把所有东西塞到一行里。实际上创建了 **4 张表**：

### checkpoints（元数据表）

```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,          -- channel_versions 等轻量元数据
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
```

### checkpoint_blobs（真正存数据的地方）

```sql
CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,                         -- 序列化的 Python 对象（pickle/msgpack）
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
```

### checkpoint_writes（中间写入记录）

```sql
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

### checkpoint_migrations（Schema 版本管理）

```sql
CREATE TABLE checkpoint_migrations (
    v INTEGER PRIMARY KEY
);
```

### 序列化方式

默认使用 `JsonPlusSerializer`，优先用 `orjson`，回退到 `msgpack`。消息列表等复杂对象序列化成二进制存到 `checkpoint_blobs` 的 `BYTEA` 列中——**对数据库完全不透明，不可 SQL 查询**。

## 2. 为什么会爆炸：O(N^2) 增长

### 问题根源

LangGraph 在**每个 super-step 边界**创建一个 checkpoint。一个简单的 `START -> A -> B -> END` 图，单次调用就产生 3 个 checkpoint。

关键在于：每个 checkpoint 包含**完整的累积消息列表**。

```
Checkpoint 1: [msg1]                        → 序列化大小 S
Checkpoint 2: [msg1, msg2]                  → 序列化大小 2S
Checkpoint 3: [msg1, msg2, msg3]            → 序列化大小 3S
...
Checkpoint N: [msg1, msg2, ..., msgN]       → 序列化大小 NS

总存储 = S + 2S + 3S + ... + NS = S × N(N+1)/2 = O(N^2)
```

### 实际数据

LangChain 官方博客（2026 年 5 月）给出的数据：
- 一个 200 轮对话的 coding agent：**5.3 GB**
- 使用 DeltaChannel 后：**129 MB**（41 倍缩减）

### PostgreSQL 特有的痛点

当序列化后的 checkpoint 超过 PostgreSQL 的 TOAST 阈值（~2KB），数据会被压缩并移到 TOAST 表。对于一个 15 步 RAG pipeline（每步 100KB state）：

| 指标 | 值 | 影响 |
|------|-----|------|
| 每步写入 | 100KB | 每次都触发 TOAST |
| 15 步运行总写入 | 1.5MB | 15 次 INSERT |
| 100 并发时 WAL 产生 | ~150 MB/s | 磁盘 I/O 饱和 |
| 复制延迟 | 3-5 秒 | 读副本不一致 |

## 3. 社区反馈

这个问题在社区中有大量讨论：

1. **langgraphjs#1138** — "How do I keep data in Postgres checkpointer database from growing unbounded?" 每次图调用产生约 100 行。
2. **LangChain Forum** — "Separate Long term memory and Checkpointing" — 每次执行产生约 50 行，请求解耦。
3. **LangChain Forum** — "How to Prune Old Messages and Blobs with PostgresSaver?" — AWS Aurora 用户请求清理方案，官方回复：自托管无内置 TTL。
4. **LangChain Forum** — "LangGraph + PostgreSQL: Chat history and summarization best practice" — 讨论双存储架构。

## 4. 解决方案全景

### 方案 A：DeltaChannel（官方，langgraph >= 1.2，Beta）

LangChain 的**官方答案**。核心思路：只存增量。

```python
from typing_extensions import Annotated
from langgraph.channels.delta import DeltaChannel

def append(state: list[str], writes: list[list[str]]) -> list[str]:
    return state + [item for batch in writes for item in batch]

class MyAgentState(TypedDict):
    items: Annotated[list[str], DeltaChannel(reducer=append, snapshot_frequency=50)]
```

工作原理：
- 普通步骤：只写**增量**（本步新增的消息）
- 每 K 步（`snapshot_frequency`，默认 1000）：写一次完整快照
- 恢复时：找最近的快照 + 回放最多 K 个增量

约束：
- Reducer 必须满足**批处理不变性**：`reducer(reducer(s, xs), ys) == reducer(s, xs + ys)`
- Beta 阶段，磁盘格式可能变化

### 方案 B：ShallowPostgresSaver（官方，稳定）

只保留**最新一个** checkpoint，不保留历史。

```python
from langgraph.checkpoint.postgres.shallow import ShallowPostgresSaver
```

PK 变为 `(thread_id, checkpoint_ns)`——每次覆盖写。丢失 time-travel 能力，但存储量骤降。

### 方案 C：消息裁剪（应用层）

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=llm,
    max_tokens=4000,
    start_on="human",
    include_system=True,
)
```

**注意**：`trim_messages` 只影响发给 LLM 的内容。要减少 checkpoint 存储，需要用 `RemoveMessage` 从 state 中真正删除消息。

### 方案 D：SummarizationMiddleware（较新 API）

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("fraction", 0.75),
            keep=("messages", 20),
        )
    ],
)
```

到达上下文窗口 75% 时触发，永久替换为 `[摘要] + [最近 20 条消息]`。减少存储但**不可逆**。

### 方案 E：Pointer State 模式（社区方案）

重量级数据（文档、嵌入向量）存 Redis/S3，State 中只保留轻量指针。继承 `PostgresSaver`，在 `put()`/`get_tuple()` 中拦截做分流。

### 方案 F：双存储架构（社区方案）

```
┌─────────────┐     ┌──────────────────┐
│   UI 显示   │────>│ 自建 messages 表  │  完整历史，供 UI 展示
└─────────────┘     └──────────────────┘

┌─────────────┐     ┌──────────────────┐
│  LLM 调用   │────>│  Checkpointer    │  只保留最近 N 条 + 摘要
└─────────────┘     └──────────────────┘
```

UI 读自己的表（快速），LLM 用 checkpointer（精简）。

## 5. 方案对比

| 方案 | 存储量级 | Time-Travel | 复杂度 | 状态 |
|------|---------|------------|--------|------|
| 默认 PostgresSaver | O(N^2) | 完整 | 低 | 稳定 |
| DeltaChannel | O(N) | 完整 | 低 | Beta |
| ShallowPostgresSaver | O(1)/thread | 无 | 低 | 稳定 |
| SummarizationMiddleware | 受控 | 部分 | 中 | 可用 |
| RemoveMessage 裁剪 | 受控 | 完整但数据丢失 | 中 | 稳定 |
| Pointer State | ~150 bytes/step | 取决于 KV store | 高 | 社区 |
| 双存储架构 | Checkpoint 保持小 | 完整 | 高 | 社区 |

## 6. 更深层的问题：两层正交问题

上面所有的存储优化（DeltaChannel、ShallowPostgresSaver）解决的都是**第一层问题**——同一份数据被重复存储 N 遍导致的 O(N^2) 膨胀。

但还有**第二层问题**：即使只存一份（Shallow），消息列表本身也在无限增长。这个问题存储方案完全不解决。

### 本质矛盾

```
对话历史：无限增长
LLM 上下文窗口：有限（128K/200K tokens）
```

到某个时间点，**必须丢信息**。问题只是"丢什么、怎么丢"：

| 策略 | LLM 能看到什么 | 丢掉了什么 | 适用场景 |
|------|---------------|-----------|---------|
| 硬裁剪 (trim_messages) | 最近 N 条完整消息 | 早期对话直接消失 | 简单聊天 |
| 摘要替换 (SummarizationMiddleware) | 摘要 + 最近 N 条 | tool_call 细节、thinking 过程、中间推理链 | 长对话助手 |
| RAG 检索 (对历史消息建索引) | 最近 N 条 + 按需检索的相关片段 | 未被检索命中的部分 | 知识密集型 agent |

### 关键认知

摘要方案下，完整的工具调用细节、thinking 过程、中间推理链**不可逆地丢失**。摘要只能保留语义层面的"做了什么、结论是什么"。

因此真实的生产架构需要三层：

```
完整历史（含 tool_call、thinking）
┌──────────────────────────┐
│     自建 messages 表      │ ← 只给 UI 用，人看
└──────────────────────────┘

精简版（摘要 + 最近 N 条）
┌──────────────────────────┐
│      Checkpointer        │ ← 给 LLM 用
└──────────────────────────┘

按需检索（向量 / 关键词索引）
┌──────────────────────────┐
│   历史消息检索引擎         │ ← 补充 LLM 遗忘的细节
└──────────────────────────┘
```

**总结：存储方案解决"同样的数据别存 N 遍"，裁剪/摘要解决"数据本身太大放不进上下文"。两个问题正交，缺一不可。**

## 7. 关键结论

1. **问题确实存在**，LangChain 官方承认并量化为 O(N^2)。
2. **不是单纯的一行**——实际是 4 张表，但 `checkpoint_blobs` 中的每个 blob 确实包含完整消息列表的序列化副本。
3. **DeltaChannel 是前进方向**，将 O(N^2) 降为 O(N)，但目前还是 Beta。
4. **自托管无内置 TTL/清理**，LangSmith 平台版有，但 OSS 用户需自己实现。
5. **生产建议**：DeltaChannel + 消息裁剪/摘要 + 独立的 UI 消息表。

## 7. 源码参考

- PostgresSaver: `libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py`
- ShallowPostgresSaver: `libs/checkpoint-postgres/langgraph/checkpoint/postgres/shallow.py`
- DeltaChannel: `langgraph/channels/delta.py`
- Base Checkpointer: `langgraph/checkpoint/base.py`
- 官方博客: [Delta Channels: Evolving our Runtime for Long-Running Agents](https://langchain.com/blog/delta-channels-evolving-agent-runtime)
