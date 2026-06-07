Now I have all the information needed. Let me write the chapter.

# 06 - 读取路径深度剖析：三信号混合检索与排序

上一章我们追踪了一条记忆从对话到持久化的完整旅程。现在轮到另一半故事：当用户调用 `search()` 时，Mem0 如何从数万条记忆中找到最相关的那几条？

答案是一套**三信号混合检索**（Hybrid Retrieval）架构。它不依赖单一的向量相似度，而是同时汇聚语义搜索、关键词匹配和实体链接三路信号，通过加法融合产生最终排序。这种设计在简单性和有效性之间取得了务实的平衡。

## search() 方法的整体流程

入口位于 `mem0/memory/main.py` 的 `search()` 方法（约第 1127 行）。完整流程分为 9 个步骤：

```
用户查询 ──> [参数校验 & 过滤器构建]
                    │
                    v
         ┌──────────────────────────┐
         │  Step 1: 查询预处理        │
         │  - BM25 词形还原           │
         │  - spaCy 实体提取          │
         └──────────┬───────────────┘
                    │
                    v
         ┌──────────────────────────┐
         │  Step 2: 查询向量化        │
         │  embed(query, "search")  │
         └──────────┬───────────────┘
                    │
        ┌───────────┼───────────────┐
        v           v               v
   ┌─────────┐ ┌─────────┐  ┌───────────┐
   │ Step 3  │ │ Step 4  │  │  Step 6   │
   │ 语义搜索 │ │ BM25搜索│  │ 实体增强   │
   │ top_k*4 │ │ top_k*4 │  │ 实体链接   │
   └────┬────┘ └────┬────┘  └─────┬─────┘
        │           │             │
        v           v             v
   ┌──────────────────────────────────┐
   │  Step 7-8: 候选集构建 & 融合排序   │
   │  score_and_rank()               │
   └──────────────┬──────────────────┘
                  │
                  v
   ┌──────────────────────────────────┐
   │  Step 9: 结果格式化               │
   │  + 可选 Reranker 二次排序         │
   └──────────────┬──────────────────┘
                  │
                  v
            返回 top_k 条结果
```

一个关键细节：语义搜索的 `internal_limit` 被设为 `max(limit * 4, 60)`。这意味着当用户请求 `top_k=20` 时，系统实际从向量存储中拉取 80 条候选，然后在融合排序中筛选到 20 条。这种**过度召回**（over-fetch）策略确保 BM25 和实体增强有足够的候选池来重新排列。

## 三路信号详解

### 信号一：语义向量相似度 [0, 1]

语义搜索是检索的主干。查询文本经过 embedding 模型向量化后，在向量存储中执行近似最近邻（ANN）搜索：

```python
# mem0/memory/main.py, _search_vector_store()
embeddings = self.embedding_model.embed(query, "search")
semantic_results = self.vector_store.search(
    query=query, vectors=embeddings, top_k=internal_limit, filters=filters
)
```

`VectorStoreBase` 的接口契约要求所有 22 种实现必须**将分数归一化到 [0, 1] 区间**，其中 1.0 表示完全匹配。这个归一化约束是混合评分正确运作的前提——如果不同向量存储返回不同量纲的分数，加法融合将毫无意义。

语义搜索擅长捕捉**意义等价**：查询 "favorite programming language" 能匹配到记忆 "prefers Python for backend development"，即使两者没有共同关键词。但它也有盲区：对专有名词、缩写和精确数值的匹配能力较弱。

### 信号二：BM25 关键词匹配 [0, 1]

BM25 弥补了语义搜索对精确词汇匹配的不足。完整管线包含两个阶段：**词形还原**和 **sigmoid 归一化**。

#### 词形还原管线

源码位于 `mem0/utils/lemmatization.py`。查询和记忆文本都经过 spaCy 的词形还原器处理：

```python
def lemmatize_for_bm25(text: str) -> str:
    nlp = get_nlp_lemma()
    if nlp is None:
        return text  # 降级到原始文本

    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)
        # 保留 -ing 形式以处理名词/动词歧义
        if token.text.endswith("ing") and token.text != lemma:
            tokens.append(token.text)
    return " ".join(tokens)
```

这个设计有一个精巧的细节：它**同时保留 -ing 原形和词元形式**。原因在于 spaCy 的词形还原是上下文相关的——"meeting" 作为名词还原为 "meeting"，作为动词还原为 "meet"。在不同的句子中，同一个词可能被还原成不同的形式。通过同时保留两者，BM25 匹配不会因为词性标注的不确定性而丢失匹配。

#### Sigmoid 归一化

原始 BM25 分数是无界的（通常在 0-20+ 之间），不能直接与 [0, 1] 的语义分数相加。Mem0 使用 logistic sigmoid 函数进行归一化：

```python
# mem0/utils/scoring.py
def normalize_bm25(raw_score, midpoint, steepness):
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))
```

sigmoid 的中点和陡度**根据查询长度自适应调整**：

```python
def get_bm25_params(query, *, lemmatized=None):
    num_terms = len(lemmatized.split()) if lemmatized else 1
    if num_terms <= 3:   return 5.0, 0.7   # 短查询：中点低、陡度高
    elif num_terms <= 6: return 7.0, 0.6
    elif num_terms <= 9: return 9.0, 0.5
    elif num_terms <= 15: return 10.0, 0.5
    else:                return 12.0, 0.5  # 长查询：中点高、陡度低
```

直觉是：短查询（如 "Python version"）的 BM25 原始分数天然较低，所以中点设为 5.0，让中等分数也能获得合理的归一化值。长查询有更多匹配词的机会，原始分数更高，所以中点上移到 12.0。这种自适应避免了短查询信号被压制或长查询信号被过度放大的问题。

目前有 15 种向量存储实现了 `keyword_search()` 方法，包括 Qdrant、Pinecone、Elasticsearch、pgvector 等。对于不支持 BM25 的存储（如 FAISS、Chroma），`keyword_search()` 返回 `None`，评分公式自动退化为仅语义搜索。

### 信号三：实体增强 [0, 0.5]

实体增强是最精细的信号，通过实体链接关系来提升相关记忆的得分。核心逻辑在 `_compute_entity_boosts()` 方法中：

```python
def _compute_entity_boosts(self, query_entities, filters):
    # 去重，最多处理 8 个实体
    deduped = []  # [(entity_type, entity_text), ...]

    # 批量嵌入所有实体文本
    embeddings = self.embedding_model.embed_batch(entity_texts, "search")

    # 并发搜索实体存储（4 线程）
    for entity_text, embedding in zip(entity_texts, embeddings):
        matches = entity_store.search(
            query=entity_text, vectors=embedding,
            top_k=500, filters=search_filters
        )
        for match in matches:
            if match.score < 0.5:  # 实体相似度门槛
                continue
            linked_memory_ids = match.payload.get("linked_memory_ids", [])
            # 链接数量衰减权重
            memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
            boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
            # 取每条记忆的最大 boost
            memory_boosts[memory_key] = max(existing, boost)
```

这里有几个值得注意的设计：

1. **独立实体存储**：实体存储在与记忆分离的向量集合 `{collection_name}_entities` 中。每个实体记录包含 `linked_memory_ids` 列表，指向它关联的所有记忆。

2. **0.5 相似度门槛**：只有与查询实体相似度 >= 0.5 的实体才参与增强，避免噪声实体的干扰。

3. **链接数量衰减**：`memory_count_weight = 1/(1 + 0.001 * (n-1)^2)` 对链接了大量记忆的高频实体（如 "Python"）施加轻微惩罚。一个链接了 1000 条记忆的实体，其权重约为 0.5，不会让所有包含 "Python" 的记忆都获得同等增强。

4. **取最大值**：当多个查询实体都增强了同一条记忆时，取最大 boost 而非累加。这防止了仅因提及多个常见实体就获得不成比例的高分。

## 评分公式

三路信号在 `mem0/utils/scoring.py` 的 `score_and_rank()` 函数中融合：

```python
ENTITY_BOOST_WEIGHT = 0.5

# 自适应分母
max_possible = 1.0                          # 语义搜索始终存在
if has_bm25:    max_possible += 1.0         # BM25 可选
if has_entity:  max_possible += ENTITY_BOOST_WEIGHT  # 实体增强可选

for result in semantic_results:
    semantic_score = result["score"]
    if semantic_score < threshold:           # 门槛过滤
        continue

    raw_combined = semantic_score + bm25_score + entity_boost
    combined = min(raw_combined / max_possible, 1.0)
```

根据激活的信号组合，`max_possible` 有四种取值：

| 激活信号 | max_possible | 场景 |
|---------|-------------|------|
| 仅语义 | 1.0 | 向量存储不支持 BM25，无 spaCy |
| 语义 + BM25 | 2.0 | 有 BM25 但无实体 |
| 语义 + 实体 | 1.5 | 无 BM25 但有实体链接 |
| 三路信号全部 | 2.5 | 完整混合检索 |

一个重要的语义：**threshold 门槛作用于语义分数而非融合分数**。这意味着即使一条记忆的 BM25 和实体增强分数很高，只要语义分数低于门槛（默认 0.1），它就会被直接丢弃。语义搜索充当了**硬门禁**，BM25 和实体增强只是在门禁之内进行重排。

`explain=True` 参数会在每条结果中附加 `score_details` 字典，暴露所有中间分数：

```python
{
    "semantic_score": 0.85,
    "bm25_score": 0.72,
    "entity_boost": 0.35,
    "raw_score": 1.92,
    "max_possible_score": 2.5,
    "final_score": 0.768,
    "threshold": 0.1
}
```

这对调试检索质量非常有用。

## Reranker 二次排序

混合评分产生的 top_k 结果可以进一步经过 Reranker 二次排序。Reranker 在 `search()` 方法的最后一步触发，仅在 `rerank=True` 且已配置 reranker 时生效：

```python
if rerank and self.reranker and original_memories:
    reranked_memories = self.reranker.rerank(query, original_memories, limit)
```

所有 Reranker 继承自 `BaseReranker`（`mem0/reranker/base.py`），接口只有一个方法：

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_k: int = None):
        """返回带有 rerank_score 字段的重排文档列表。"""
```

### 5 种 Reranker 实现

| Provider | 模型示例 | 特点 | 适用场景 |
|----------|---------|------|---------|
| **Cohere** | `rerank-v3.5` | API 调用，支持 `max_chunks_per_doc` | 生产环境，高质量排序 |
| **HuggingFace** | `BAAI/bge-reranker-base` | 本地 cross-encoder，批处理推理 | 私有部署，无外部 API 依赖 |
| **SentenceTransformer** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 基于 `sentence-transformers` 的 CrossEncoder | 轻量级本地排序 |
| **LLM** | `gpt-4o-mini` | 通用 LLM 逐文档打分（0-1） | 灵活，可利用已有 LLM 配置 |
| **ZeroEntropy** | `zerank-1` | API 调用，专用排序模型 | 第三方专业排序服务 |

**Cross-encoder vs Bi-encoder**：混合检索的语义搜索使用 bi-encoder（查询和文档独立编码），速度快但精度有限。Reranker 中的 HuggingFace 和 SentenceTransformer 使用 cross-encoder（查询和文档联合编码），精度更高但只能处理少量候选。这是经典的**粗排 + 精排**两阶段架构。

**LLM Reranker 的安全设计**：LLM Reranker 将评分指令放在 system message 中，将用户查询和文档放在 user message 中，并截断输入到 4000 字符，防止 prompt injection。它还使用正则表达式从 LLM 输出中提取分数，解析失败时默认返回 0.5。

所有 Reranker 都实现了**故障降级**：如果重排失败，静默回退到原始排序并赋予默认 `rerank_score = 0.0`。

## 元数据过滤操作符

`search()` 的 `filters` 参数不仅用于身份作用域（`user_id`/`agent_id`/`run_id`），还支持丰富的元数据过滤。过滤在向量存储层执行，先于混合评分：

```python
# 精确匹配
m.search("query", filters={"user_id": "u1", "category": "work"})

# 比较操作符
m.search("query", filters={"user_id": "u1", "priority": {"gte": 5}})

# 集合操作符
m.search("query", filters={"user_id": "u1", "tag": {"in": ["health", "fitness"]}})

# 逻辑组合
m.search("query", filters={
    "user_id": "u1",
    "OR": [
        {"category": "work"},
        {"priority": {"gte": 8}}
    ]
})
```

完整操作符列表：`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`, `contains`, `icontains`, `*`（通配符），以及逻辑组合 `AND`, `OR`, `NOT`。

过滤器在 `_process_metadata_filters()` 中被转换为统一的内部格式（如 `$or`, `$not`），然后由各向量存储自行翻译为原生查询语法。这种两阶段转换确保了上层 API 的一致性，同时允许不同存储使用各自最高效的过滤实现。

## top_k 和 threshold 的调优策略

这两个参数直接影响检索的**精确率-召回率平衡**：

- **threshold**（默认 0.1）：语义相似度的硬门槛。提高到 0.3-0.5 可以显著减少不相关结果（提高精确率），但可能丢失措辞差异大的相关记忆（降低召回率）。在记忆数量大（>10000）且噪声多时，建议提高此值。

- **top_k**（默认 20）：返回的最大结果数。注意内部 over-fetch 倍率是 4x，所以 `top_k=100` 会在向量存储中检索 400 条候选。对于延迟敏感的场景，应控制 `top_k` 在合理范围内。

一个常见陷阱：`threshold` 过低（如默认的 0.1）在大记忆库中会导致大量低质量候选进入混合评分，增加 BM25 和实体增强的计算开销。但 `threshold` 过高又会在 BM25 能拯救的场景中过早丢弃候选。0.1 的默认值是一个宽松的起点，鼓励用户根据实际数据调整。

## 设计思考

### 为什么选择加法融合而非学习融合？

加法融合（additive scoring）是最简单的多信号组合策略。更复杂的替代方案包括：

- **学习融合（learned fusion）**：训练一个模型来学习三路信号的最优权重。需要标注数据和训练基础设施。
- **倒数排名融合（RRF）**：基于排名位置而非分数的融合，对分数分布不敏感。
- **加权融合**：为每路信号分配可调权重。

Mem0 选择加法融合的理由很务实：**不需要训练数据，不需要调参，不需要额外基础设施**。对于一个面向 10 万+ 开发者的开源框架，复杂度的下限至关重要。自适应分母 `max_possible` 优雅地处理了信号缺失的情况——当向量存储不支持 BM25 时，公式自动退化而无需特殊逻辑。

代价是：信号权重是硬编码的（语义和 BM25 权重均为 1.0，实体增强权重为 0.5），无法根据具体应用场景调整。对于大多数记忆检索场景，语义搜索作为主信号的优先级高于关键词匹配是合理的，但某些精确匹配优先的场景（如代码片段检索）可能需要不同的权重分配。

### 实体增强权重 0.5 的上限从何而来？

`ENTITY_BOOST_WEIGHT = 0.5` 这个常量设计了实体信号的上限。在最理想的情况下（实体相似度 = 1.0，链接数量权重 = 1.0），实体增强最多为 0.5 分。与语义和 BM25 各自最高 1.0 分相比，实体增强被定位为**辅助信号而非主导信号**。

设定为 0.5 而非 1.0 有两个理由：

1. **实体链接是间接信号**：它不直接衡量查询与记忆的相关性，而是通过共享实体建立间接关联。一条记忆提及 "Python" 不意味着它与所有关于 Python 的查询都相关。
2. **防止实体过度主导**：如果权重为 1.0，一条语义分数中等但实体匹配完美的记忆可能跳到语义分数高的记忆前面，这通常不是用户期望的行为。

链接数量衰减公式 `1/(1 + 0.001 * (n-1)^2)` 进一步细化了这一设计：链接了 1 条记忆的实体获得满权重，链接了 100 条的权重约为 0.91，链接了 1000 条的约为 0.50。这种**温和的平方衰减**既不会让高频实体完全失效，也避免了它们对大量记忆施加等量增强。

> **下一章**: [07 - 图记忆与平台能力：OSS 之外的世界](07-graph-memory-and-platform.md)