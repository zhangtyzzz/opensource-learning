Now I have all the data needed. Let me write the chapter.

# 05 - 写入路径深度剖析：从对话到记忆的 8 阶段流水线

本章是全书技术密度最高的章节。我们将逐行走读 `Memory.add()` 方法内部的 V3 批处理流水线，从一段对话文本进入系统，到一条条精炼的记忆被持久化到向量存储，再到实体链接建立关联——完整 8 个阶段，不遗漏任何关键实现细节。

## add() 方法：写入路径的入口

所有写入操作始于 `Memory.add()`（`mem0/memory/main.py` 第 574 行）。该方法接受以下关键参数：

```python
def add(
    self,
    messages,           # str | dict | list[dict] — 对话消息
    *,
    user_id=None,       # 三级身份作用域
    agent_id=None,
    run_id=None,
    metadata=None,      # 自定义元数据
    infer: bool = True, # 关键开关：LLM 提取 vs 原始存储
    memory_type=None,   # "procedural_memory" 或默认
    prompt=None,        # 自定义提取提示词
):
```

`infer` 参数决定了两条完全不同的写入路径：

- **`infer=True`（默认）**：进入 V3 8 阶段流水线，调用 LLM 提取结构化事实
- **`infer=False`**：跳过 LLM，将原始消息直接作为记忆存储，每条非 system 消息对应一条记忆

`infer=False` 的实现极为简洁——遍历消息列表，逐条嵌入并写入向量存储（第 664-698 行）。真正复杂的逻辑全部在 `infer=True` 路径中。

## V3 8 阶段流水线全景

以下是完整流水线的架构示意图：

```
输入: messages + filters(user_id/agent_id/run_id)
  │
  ▼
┌─────────────────────────────────────────────┐
│ Phase 0: 上下文收集                          │
│   构建 session_scope → 获取最近 10 条消息     │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 1: 现有记忆检索                        │
│   嵌入当前消息 → 向量检索 top-10 相关记忆     │
│   UUID → 整数映射（防幻觉）                   │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 2: LLM 提取（单次调用）                │
│   system: ADDITIVE_EXTRACTION_PROMPT (940行) │
│   user: 组装的上下文提示词                    │
│   → 返回 JSON: {"memory": [...]}            │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 3: 批量嵌入                            │
│   embed_batch() 一次性嵌入所有提取文本        │
│   失败降级为逐条嵌入                          │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 4-5: CPU 处理 + 哈希去重               │
│   MD5(text) → 与已有哈希 + 批内哈希比对      │
│   构建 BM25 词元化文本                        │
│   组装 payload 元数据                         │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 6: 批量持久化                          │
│   vector_store.insert() 批量写入             │
│   SQLite history 表批量写入 ADD 事件          │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 7: 批量实体链接                        │
│   spaCy 批量实体提取 → 全局去重              │
│   批量嵌入 → 批量搜索已有实体                │
│   ≥0.95 相似度: 更新 linked_memory_ids       │
│   <0.95: 批量插入新实体                      │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│ Phase 8: 消息保存 + 返回结果                  │
│   save_messages() → 保留最近 10 条            │
│   返回 [{"id": ..., "memory": ..., "event": "ADD"}] │
└─────────────────────────────────────────────┘
```

## Phase 0：上下文收集

```python
# Phase 0: Context gathering
session_scope = _build_session_scope(filters)
last_messages = self.db.get_last_messages(session_scope, limit=10)
parsed_messages = parse_messages(messages)
```

`_build_session_scope()` 将身份过滤器拼成确定性字符串（如 `"agent_id=bot1&user_id=alice"`），用作 SQLite `messages` 表的分区键。`get_last_messages()` 返回该 scope 下最近 10 条消息，供后续提示词中的 "Last k Messages" 段落使用——这为 LLM 提供了代词消解所需的对话上下文。

## Phase 1：现有记忆检索

```python
# Phase 1: Existing memory retrieval
search_filters = {k: v for k, v in filters.items()
                  if k in ("user_id", "agent_id", "run_id") and v}
query_embedding = self.embedding_model.embed(parsed_messages, "search")
existing_results = self.vector_store.search(
    query=parsed_messages, vectors=query_embedding,
    top_k=10, filters=search_filters,
)

# Map UUIDs to integers (anti-hallucination)
uuid_mapping = {}
for idx, mem in enumerate(existing_results):
    uuid_mapping[str(idx)] = mem.id
    existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})
```

这里有一个精妙的防幻觉设计：将 UUID 映射为连续整数（"0", "1", "2"...）。LLM 在生成 `linked_memory_ids` 时使用这些短 ID，避免了让 LLM 复制长 UUID 字符串时产生的编造风险。

## Phase 2：LLM 提取——940 行提示词的设计哲学

Phase 2 是整个流水线的智慧核心。系统向 LLM 发出一次调用，使用 `ADDITIVE_EXTRACTION_PROMPT`（约 940 行）作为系统提示词，`generate_additive_extraction_prompt()` 组装的上下文作为用户提示词。

### 提示词的输入结构

用户提示词由 `generate_additive_extraction_prompt()` 组装（第 1016 行），包含七个段落：

| 段落 | 内容 | 用途 |
|------|------|------|
| Summary | 用户画像摘要 | 为新用户提供已建立的上下文 |
| Last k Messages | 最近 10 条历史消息 | 代词消解和上下文延续 |
| Recently Extracted | 本会话已提取的记忆 | 会话内去重 |
| Existing Memories | 向量检索到的 top-10 记忆 | 跨会话去重 + 链接 |
| New Messages | 当前要处理的对话 | 提取来源 |
| Observation Date | 对话发生日期 | 时间引用锚点 |
| Current Date | 系统当前日期 | 区分于 Observation Date |

### 三个核心设计决策

**1. 双向提取：从用户和助手消息同时提取**

提示词明确要求从两个角色提取：用户消息提供个人事实、偏好、计划；助手消息提供推荐、创建的计划、研究的信息。但对助手消息施加严格约束——只提取"真正新颖"的内容。

**2. 防止"回声提取"**

这是提示词中反复强调的规则（No Echo Extraction）：当助手消息只是重述、总结或确认用户已说的内容时，不得重复提取。例如用户说"我想要每天 7:30 的签到"，助手回复"已设置每天 7:30 的签到"——只从用户消息提取一次。但单条助手消息可能同时包含回声和新信息，提示词要求跳过回声部分但仍提取新事实。

**3. 时间引用解析**

提示词区分 Observation Date（对话发生时间）和 Current Date（系统当前时间），所有相对时间引用（"昨天"、"上周"、"下个月"）必须基于 Observation Date 解析为绝对日期。这确保了"上周去了巴黎"在 6 个月后仍然有意义。

### ADD-only 输出格式

V3 提示词只允许一种操作：ADD。输出格式为：

```json
{
  "memory": [
    {
      "id": "0",
      "text": "User's name is Marcus...",
      "attributed_to": "user",
      "linked_memory_ids": ["uuid-of-related-memory"]
    }
  ]
}
```

每条记忆包含四个字段：顺序 `id`、事实文本 `text`、归属方 `attributed_to`（"user" 或 "assistant"）、以及可选的 `linked_memory_ids`（引用已有记忆的 UUID）。

## Phase 3：批量嵌入

```python
mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
embed_map = dict(zip(mem_texts, mem_embeddings_list))
```

所有提取的记忆文本通过 `embed_batch()` 一次性嵌入，避免逐条调用的 N 次网络往返。若批量嵌入失败，降级为逐条嵌入，体现了"批量优先、逐条兜底"的容错策略。

## Phase 4-5：CPU 处理与 MD5 哈希去重

```python
# 收集已有记忆的哈希值
existing_hashes = set()
for mem in existing_results:
    h = mem.payload.get("hash")
    if h:
        existing_hashes.add(h)

seen_hashes = set()  # 批内去重
for mem in extracted_memories:
    text = mem.get("text")
    mem_hash = hashlib.md5(text.encode()).hexdigest()
    if mem_hash in existing_hashes or mem_hash in seen_hashes:
        continue  # 跳过重复
    seen_hashes.add(mem_hash)
    
    text_lemmatized = lemmatize_for_bm25(text)
    # 组装 payload: data, text_lemmatized, hash, created_at, updated_at, attributed_to
```

去重发生在两个层面：**跨会话**（与 Phase 1 检索到的已有记忆哈希比对）和**批内**（同一批提取结果中的重复）。每条记忆还经过 BM25 词元化处理（`lemmatize_for_bm25`），为读取路径的关键词搜索做准备。

## Phase 6：批量持久化

```python
# 批量写入向量存储
self.vector_store.insert(
    vectors=all_vectors, ids=all_ids, payloads=all_payloads,
)

# 批量写入历史记录
history_records = [
    {"memory_id": r[0], "old_memory": None, "new_memory": r[1],
     "event": "ADD", "created_at": r[3].get("created_at")}
    for r in records
]
self.db.batch_add_history(history_records)
```

向量存储和 SQLite history 表的写入都采用批量操作。两者都有逐条插入的降级路径。history 表记录每次变更的完整审计信息：`memory_id`、`old_memory`（ADD 时为 None）、`new_memory`、`event` 类型、时间戳。

## Phase 7：批量实体链接——spaCy 四种实体类型

Phase 7 是 V3 流水线中最精密的阶段，分五个子步骤：

**7a. spaCy 批量提取 + 全局去重**

```python
all_entities = extract_entities_batch(all_texts)
global_entities = {}  # normalized_key -> [entity_type, entity_text, {memory_ids}]
```

`extract_entities()`（`mem0/utils/entity_extraction.py`）使用 spaCy NLP 管道提取四种实体类型：

| 类型 | 识别方式 | 示例 |
|------|---------|------|
| PROPER | 句中大写多词序列（排除句首） | "Osteria Francescana", "Shopify" |
| QUOTED | 引号内文本（单引号或双引号） | "The Last Dance" |
| COMPOUND | 名词-名词复合短语 | "machine learning", "aerial yoga" |
| NOUN | 单名词兜底（当复合修饰语过于笼统时） | "hiking", "pottery" |

实体提取维护了大量过滤列表：`_GENERIC_HEADS`（排除"thing"、"stuff"等泛化词）、`_CIRCUMSTANTIAL_MODS`（排除"solo"、"first"等环境修饰语）、`_NON_SPECIFIC_ADJ`（排除"good"、"new"等非特异性形容词），确保提取的实体具有检索价值。

**7b-7e. 嵌入、搜索、更新或插入**

全局去重后的实体被批量嵌入，然后在独立的实体集合（`{collection_name}_entities`）中批量搜索。相似度 >= 0.95 的匹配项更新其 `linked_memory_ids` 列表；低于阈值的作为新实体批量插入。

实体存储的 payload 结构：

```json
{
  "data": "Shopify",
  "entity_type": "PROPER",
  "linked_memory_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "user_id": "alice"
}
```

## Phase 8：消息保存与返回

```python
self.db.save_messages(messages, session_scope)
```

原始对话消息被保存到 SQLite `messages` 表，每个 `session_scope` 保留最近 10 条（超出的自动淘汰）。这些消息在下次 `add()` 调用时作为 Phase 0 的 "Last k Messages" 上下文。

## SQLiteManager：history 表与 messages 表

`SQLiteManager`（`mem0/memory/storage.py`）管理两张表：

```sql
-- 记忆变更日志：每次 ADD/UPDATE/DELETE 写一行
CREATE TABLE history (
    id           TEXT PRIMARY KEY,  -- UUID
    memory_id    TEXT,              -- 关联的记忆 ID
    old_memory   TEXT,              -- 变更前内容（ADD 时为 NULL）
    new_memory   TEXT,              -- 变更后内容
    event        TEXT,              -- "ADD" | "UPDATE" | "DELETE"
    created_at   DATETIME,
    updated_at   DATETIME,
    is_deleted   INTEGER,
    actor_id     TEXT,              -- 操作者标识
    role         TEXT               -- "user" | "assistant"
);

-- 最近消息窗口：每个 session_scope 保留最近 10 条
CREATE TABLE messages (
    id            TEXT PRIMARY KEY,
    session_scope TEXT,             -- "agent_id=bot1&user_id=alice"
    role          TEXT,
    content       TEXT,
    name          TEXT,
    created_at    DATETIME
);
```

`messages` 表的淘汰策略值得注意：每次 `save_messages()` 后执行 `DELETE WHERE id NOT IN (SELECT id FROM messages WHERE session_scope = ? ORDER BY created_at DESC LIMIT 10)`，确保窗口不超过 10 条。

## 设计思考

### 为什么 V3 选择 ADD-only 而非 ADD/UPDATE/DELETE？

V2 的提取提示词要求 LLM 同时判断四种操作（ADD/UPDATE/DELETE/NONE），这给 LLM 施加了沉重的推理负担：它需要理解新信息、比较已有记忆、判断冲突关系、决定操作类型。在实践中，LLM 经常在 UPDATE 和 DELETE 决策上产生幻觉——错误地删除有效记忆，或将不相关的记忆标记为需要更新。

V3 的 ADD-only 策略大幅简化了任务：LLM 只需要从对话中提取事实，不需要做任何关于已有记忆的修改决策。去重交给确定性的 MD5 哈希比对，关联通过 `linked_memory_ids` 建立软链接。这是一个"宁可多存、不可误删"的保守策略。

代价是什么？矛盾的记忆会累积。如果用户先说"我喜欢咖啡"后说"我现在更喜欢茶了"，两条记忆都会存在，只通过 `linked_memory_ids` 关联。冲突解决被推迟到读取时（由消费记忆的应用层处理），而非在写入时解决。

### MD5 哈希去重的能力与盲区

MD5 哈希去重是精确匹配：只有文本完全相同时才去重。这意味着"User likes coffee"和"User enjoys coffee"会被视为两条不同的记忆。语义近似重复是 V3 pipeline 的已知盲区——提示词中虽然有 "No Within-Response Duplication" 规则要求 LLM 自查，但这依赖 LLM 的自律而非系统保证。

更深层的限制是：哈希比对只覆盖 Phase 1 检索到的 top-10 记忆。如果一条语义重复的记忆排在第 11 位，哈希去重无法发现它。这是一个在记忆量增长后可能恶化的问题。

### 实体链接的 0.95 阈值

Phase 7 中实体匹配使用 0.95 的高相似度阈值。这是一个偏保守的选择——宁可创建一个新实体，也不错误合并两个不同实体。例如"machine learning"和"deep learning"虽然相关但不应合并为同一实体。0.95 的阈值基本只允许拼写变体级别的匹配。

> **下一章**: [06 - 读取路径深度剖析：三信号混合检索与排序](06-read-path.md)