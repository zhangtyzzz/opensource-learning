Now I have all the source code I need. Let me write the chapter.

# 07 - 检索增强生成：从数据加载到语义检索的全链路

RAG（Retrieval-Augmented Generation）是 LangChain 最高频的应用模式。它的核心思路简单直接：先从外部数据源中检索与用户查询相关的文档片段，再将这些片段作为上下文注入 Prompt，让 LLM 基于真实数据生成回答。但"简单"的思路背后是一条相当长的数据处理管线，LangChain 为这条管线上的每个环节都设计了独立的抽象接口。

本章将沿着数据流方向，逐层解析 RAG 全链路中的六个核心抽象。

## RAG 数据流全景

```
                         RAG Pipeline
                         
 [原始数据源]                                      [用户查询]
      |                                                |
      v                                                v
 +-----------+    +-------------+    +------------+    +-----------+
 | BaseLoader | -> | TextSplitter | -> | Embeddings | -> | VectorStore|
 | 文档加载   |    | 文本切分     |    | 向量化      |    | 向量存储   |
 +-----------+    +-------------+    +------------+    +-----------+
      |                                                     |
      v                                                     v
 [Document]       [Document chunks]  [float vectors]   as_retriever()
                                                            |
                                                            v
                                                    +--------------+
                                                    | BaseRetriever |
                                                    | 检索器        |
                                                    +--------------+
                                                            |
                                                            v
                                                    [list[Document]]
                                                            |
                                                            v
                                                    +---------------+
                                                    | Compressor /   |
                                                    | Transformer    |
                                                    | 后处理         |
                                                    +---------------+
                                                            |
                                                            v
                                                    [精炼的 Documents] -> Prompt -> LLM
```

## Document 数据模型：刻意与消息系统分离

RAG 管线中流动的核心数据类型是 `Document`，定义在 `libs/core/langchain_core/documents/base.py`。

```python
class BaseMedia(Serializable):
    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    metadata: dict = Field(default_factory=dict)

class Document(BaseMedia):
    page_content: str
    type: Literal["Document"] = "Document"
```

`Document` 的设计极为克制：只有 `page_content`（文本内容）、`metadata`（任意键值对元数据）和可选的 `id` 三个字段。源码注释中多次强调：

> "Document is for **retrieval workflows**, not chat I/O. For sending text to an LLM in a conversation, use message types from `langchain.messages`."

这种分离是有意为之的。`Document` 模块的 `__init__.py` 文件开头用一个完整的对比表解释了这一区分：

- **Document**（本模块）：用于数据检索与处理流程——向量存储、检索器、RAG 管线、文本切分、嵌入和语义搜索
- **Content Block**（`messages.content`）：用于 LLM 对话 I/O——多模态消息内容、工具调用、推理过程、引用

这种分离让 `Document` 可以保持轻量（没有 content block 的复杂类型系统），而 metadata 字段可以自由携带来源信息（`source`、`page`、`start_index` 等）而不受消息格式约束。`BaseMedia` 基类还派生出 `Blob`，灵感来自浏览器的 [Mozilla Blob API](https://developer.mozilla.org/en-US/docs/Web/API/Blob)，用于表示原始二进制数据（文件内容、字节流），与 Document 解析器配合使用。

## BaseLoader：文档加载器抽象

`BaseLoader`（`libs/core/langchain_core/document_loaders/base.py`）定义了数据进入 RAG 系统的入口。

```python
class BaseLoader(ABC):
    def load(self) -> list[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        # 子类应该实现这个方法
        ...

    async def alazy_load(self) -> AsyncIterator[Document]:
        # 默认实现：在线程池中调用同步版本
        ...
```

设计要点值得注意：`load()` 方法不是抽象方法，而是一个便利方法，它的实现只是 `list(self.lazy_load())`。真正需要子类实现的是 `lazy_load()`——一个生成器方法。这个设计决策反映了对大规模数据加载的考量：逐条 yield Document，而非一次性加载到内存。

`BaseLoader` 还附带了一个 `load_and_split()` 方法，直接将加载和切分组合在一起。但源码中用 `danger` 标记将这个方法标注为"应被视为已弃用"——LangChain 团队认为加载和切分应该是独立的步骤，组合它们违反了单一职责原则。

同模块中还定义了 `BaseBlobParser`，它接受 `Blob` 对象并输出 `Document` 序列，实现了加载（获取原始字节）与解析（提取结构化文本）的解耦。

## TextSplitter 体系：结构化文本切分

文本切分是 RAG 中最容易被低估的环节。`TextSplitter` 的基类定义在独立包 `langchain-text-splitters`（`libs/text-splitters/langchain_text_splitters/base.py`）中，它本身继承自 `BaseDocumentTransformer`：

```python
class TextSplitter(BaseDocumentTransformer, ABC):
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None: ...

    @abstractmethod
    def split_text(self, text: str) -> list[str]: ...

    def split_documents(self, documents: Iterable[Document]) -> list[Document]: ...
```

`TextSplitter` 继承 `BaseDocumentTransformer` 意味着它既可以独立使用（`split_text`），也可以作为文档转换管线的一部分（`transform_documents`）。

`_merge_splits` 方法实现了核心的合并逻辑：将小片段合并为不超过 `chunk_size` 的大块，同时保证相邻块之间有 `chunk_overlap` 的重叠，确保语义连续性不被硬切断。

### 切分器家族

`langchain-text-splitters` 包中包含丰富的切分器实现：

| 切分器 | 源码文件 | 切分策略 |
|--------|----------|----------|
| `CharacterTextSplitter` | `character.py` | 按单一分隔符（默认 `\n\n`）切分 |
| `RecursiveCharacterTextSplitter` | `character.py` | **递归**尝试多级分隔符 `["\n\n", "\n", " ", ""]` |
| `MarkdownTextSplitter` | `markdown.py` | 按 Markdown 标题层级切分 |
| `HTMLSectionSplitter` | `html.py` | 按 HTML 标签结构切分 |
| `PythonCodeTextSplitter` | `python.py` | 按 Python 类/函数定义切分 |
| `TokenTextSplitter` | `base.py` | 按 tiktoken token 数切分 |

`RecursiveCharacterTextSplitter` 是实践中最常用的默认选择。它的核心算法是递归降级：先尝试用最粗粒度的分隔符（`\n\n`，段落边界）切分，如果某个片段仍然超过 `chunk_size`，就降级到下一级分隔符（`\n`，行边界），以此类推直到单字符。

它还通过 `from_language()` 类方法支持编程语言感知的切分。例如 Python 的分隔符优先级是 `["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]`——先尝试在类和函数定义边界切分，保持代码块的语义完整性。

## Embeddings 接口：文档向量化与查询向量化

`Embeddings`（`libs/core/langchain_core/embeddings/embeddings.py`）是 RAG 中最精简的抽象之一：

```python
class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed_query(self, text: str) -> list[float]: ...
```

为什么要区分 `embed_documents` 和 `embed_query`？源码注释解释道："Usually the query embedding is identical to the document embedding, but the abstraction allows treating them independently." 某些嵌入模型（如 Instructor 系列）确实会为文档和查询使用不同的前缀指令，产生不同的嵌入向量。`embed_documents` 接受批量输入，`embed_query` 接受单条输入，这种签名差异也反映了典型的使用模式：索引时批量处理文档，检索时逐条处理查询。

注意 `Embeddings` 没有继承 `Runnable`——它是一个纯接口 ABC，不参与 LCEL 管道组合。这是因为嵌入操作通常被封装在 `VectorStore` 内部调用，而非作为独立的管道步骤。

## VectorStore：向量存储与相似性搜索

`VectorStore`（`libs/core/langchain_core/vectorstores/base.py`）是 RAG 的核心存储层：

```python
class VectorStore(ABC):
    def add_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None,
                  *, ids: list[str] | None = None, **kwargs) -> list[str]: ...
    def add_documents(self, documents: list[Document], **kwargs) -> list[str]: ...

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]: ...

    def similarity_search_with_score(self, *args, **kwargs) -> list[tuple[Document, float]]: ...
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4,
                                                 **kwargs) -> list[tuple[Document, float]]: ...
    def max_marginal_relevance_search(self, query: str, k: int = 4,
                                       fetch_k: int = 20, lambda_mult: float = 0.5,
                                       **kwargs) -> list[Document]: ...

    @classmethod
    @abstractmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings, ...) -> Self: ...

    def as_retriever(self, **kwargs) -> VectorStoreRetriever: ...
```

`VectorStore` 同样没有继承 `Runnable`。它提供三种搜索模式：
- `similarity`：纯相似度搜索
- `similarity_score_threshold`：带分数阈值过滤的相似度搜索
- `mmr`（Maximal Marginal Relevance）：在相关性和多样性之间取平衡

源码中还包含三个静态的相关度评分归一化函数（欧氏距离、余弦距离、最大内积），将不同的距离度量统一映射到 `[0, 1]` 范围，`0` 表示不相似，`1` 表示最相似。

`as_retriever()` 方法是关键的桥接点——它将 `VectorStore` 包装为一个 `VectorStoreRetriever`，从而接入 Runnable 体系。

## BaseRetriever：比 VectorStore 更通用的抽象

`BaseRetriever`（`libs/core/langchain_core/retrievers.py`）是 RAG 查询侧的核心抽象：

```python
RetrieverInput = str
RetrieverOutput = list[Document]

class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]: ...
```

类型签名 `RunnableSerializable[str, list[Document]]` 清晰地表达了 Retriever 的本质：一个接受字符串查询、返回文档列表的 Runnable。这意味着它可以直接参与 LCEL 管道组合：

```python
chain = retriever | prompt | model | parser
```

`BaseRetriever` 的 `invoke` 方法内部做了三件事：配置 CallbackManager、调用 `_get_relevant_documents`、触发追踪事件（`on_retriever_start` / `on_retriever_end` / `on_retriever_error`）。子类只需实现 `_get_relevant_documents`，就自动获得回调追踪、LangSmith 集成和完整的 Runnable 能力。

`VectorStoreRetriever` 是 `BaseRetriever` 的内置实现，它持有一个 `VectorStore` 引用和 `search_type` / `search_kwargs` 配置，在 `_get_relevant_documents` 中委托给 `VectorStore` 的对应搜索方法。

## Document Compressor / Transformer：检索后处理

检索到的文档通常需要后处理——过滤不相关内容、重新排序、压缩长文本。LangChain 提供了两个互补的抽象：

```python
# libs/core/langchain_core/documents/compressor.py
class BaseDocumentCompressor(BaseModel, ABC):
    @abstractmethod
    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]: ...

# libs/core/langchain_core/documents/transformers.py
class BaseDocumentTransformer(ABC):
    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]: ...
```

`BaseDocumentCompressor` 接收查询上下文（`query` 参数），可以基于查询对文档进行针对性压缩——例如使用 LLM 提取与查询最相关的段落。`BaseDocumentTransformer` 则是查询无关的通用转换——例如基于嵌入相似度的去重过滤。值得注意的是，`TextSplitter` 正是继承自 `BaseDocumentTransformer`。

源码中 `BaseDocumentCompressor` 的注释甚至建议："Users should favor using a `RunnableLambda` instead of sub-classing from this interface"——团队在反思这个抽象是否仍然必要，暗示未来可能简化为纯 Runnable 组合。

## Indexing API：解决增量更新问题

当数据源持续更新时，朴素地重新索引全部文档会导致重复和浪费。Indexing API（`libs/core/langchain_core/indexing/`）通过 `RecordManager` 解决这个问题。

```python
def index(
    docs_source: BaseLoader | Iterable[Document],
    record_manager: RecordManager,
    vector_store: VectorStore | DocumentIndex,
    *,
    cleanup: Literal["incremental", "full", "scoped_full"] | None = None,
    source_id_key: str | Callable[[Document], str] | None = None,
    key_encoder: Literal["sha1", "sha256", "sha512", "blake2b"] | Callable[[Document], str] = "sha1",
) -> IndexingResult: ...
```

核心机制：对每个 Document 的 `page_content` 和 `metadata` 计算哈希值作为唯一标识（`_get_document_with_hash`），然后通过 `RecordManager` 记录哪些文档已被索引、何时索引的。下次索引时，哈希未变的文档被跳过，哈希变化的文档被更新，不再出现的文档被清理。

`RecordManager` 是一个时间戳集合抽象，记录每个文档哈希的写入时间和来源 ID。它的 `get_time()` 方法被设计为必须从服务端获取——源码注释强调"It's important to get this from the server to ensure a monotonic clock, otherwise there may be data loss when cleaning up old documents"。这是因为增量清理依赖时间戳比较，客户端时钟漂移会导致误删。

三种清理模式各有适用场景：
- `incremental`：实时清理过期文档，减少用户看到重复内容的概率，但要求 `source_id_key`
- `full`：索引完成后删除所有未出现的文档——要求数据源返回完整数据集
- `scoped_full`：折中方案，只删除本次涉及的 source ID 范围内的过期文档

## 设计思考

### 为什么 BaseRetriever 是比 VectorStore 更通用的抽象层

`VectorStore` 代表一种特定的检索实现——基于嵌入向量的相似性搜索。但并非所有文档检索都需要向量。源码文件头部的注释直接点明了这一点：

> "A retriever does not need to be able to store documents, only to return (or retrieve) it. Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well."

`BaseRetriever` 作为 `Runnable[str, list[Document]]`，可以包装任何检索逻辑：TF-IDF 搜索、BM25 全文检索、知识图谱查询、SQL 数据库查询、甚至对外部 API 的调用。源码中的示例就展示了一个基于 scikit-learn TF-IDF 的 Retriever。

这种分层设计让 VectorStore 专注于存储和向量搜索的具体实现，而 Retriever 作为 LCEL 管道中的标准组件提供统一的 Runnable 接口。`as_retriever()` 桥接方法让两层之间的转换几乎无摩擦。

这也解释了为什么 `VectorStore` 没有继承 `Runnable` 而 `BaseRetriever` 继承了——VectorStore 的接口（`add_texts`、`similarity_search_by_vector` 等）太过具体化，无法用 `Runnable[str, list[Document]]` 的通用签名覆盖。Retriever 的"只检索不存储"的简单契约反而更适合作为管道组件。

### Indexing API 如何解决增量更新问题

增量更新表面上简单（比较新旧文档、更新差异），但在分布式系统中充满陷阱。Indexing API 的设计做了几个关键选择：

第一，基于内容哈希而非文档 ID 判断变更。`_get_document_with_hash` 同时哈希 `page_content` 和 `metadata`，这意味着即使文档来源相同，内容或元数据的任何变化都会触发重新索引。

第二，RecordManager 与 VectorStore 分离部署。源码注释坦承这带来了一致性风险："The record manager is currently implemented separately from the vectorstore, which means that the overall system becomes distributed and may create issues with consistency." 写入顺序是先写 VectorStore 再写 RecordManager——如果 VectorStore 写入成功但 RecordManager 更新失败，下次索引时文档会被重新写入（幂等），而不是丢失。

第三，`scoped_full` 模式（v0.3.25 新增）是对 `full` 和 `incremental` 的务实折中。当数据源无法一次返回全量数据集，又不想承担 `incremental` 模式的冗余计算时，`scoped_full` 只在本批次涉及的 source ID 范围内做全量清理，在内存中追踪已见过的 source ID。

> **下一章**: [08 - Agent 系统：从链式调用到图状态机](08-agents-and-orchestration.md)