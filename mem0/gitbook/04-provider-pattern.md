Now I have all the information needed. Let me write the chapter.

# 04 - Provider 插件体系：55+ 实现的统一抽象

如果要在 Mem0 的代码库中选出一个最值得学习的架构设计，Provider 插件体系当之无愧。它用四个抽象基类和四个工厂类，统一管理了 55+ 个具体实现——覆盖 LLM、Embedding、向量存储和 Reranker 四个维度。任何用户都可以通过修改一行配置字符串来切换底层基础设施，而上层的记忆逻辑完全不感知变化。

本章将从抽象基类的接口契约出发，剖析 Factory 的延迟导入机制，然后以 Qdrant 向量存储和 OpenAI LLM 为案例完整走读实现过程，最后讨论如何添加自定义 Provider。

## 四维 Provider 全景

```
                        Memory 编排器 (memory/main.py)
                               │
            ┌──────────┬───────┴───────┬──────────┐
            ▼          ▼               ▼          ▼
        LlmFactory  EmbedderFactory  VectorStore  RerankerFactory
            │          │            Factory        │
            ▼          ▼               ▼          ▼
       ┌─────────┐ ┌────────┐   ┌──────────┐  ┌────────┐
       │ LLMBase │ │Embedding│  │VectorStore│  │ Base   │
       │  (ABC)  │ │Base(ABC)│  │Base (ABC) │  │Reranker│
       └────┬────┘ └───┬────┘  └─────┬─────┘  │ (ABC)  │
            │          │             │         └───┬────┘
    ┌───┬───┼───┐  ┌───┼───┐  ┌──┬──┼──┬──┐   ┌──┼──┐
    │   │   │   │  │   │   │  │  │  │  │  │   │  │  │
   OAI Ant Gem ..  OAI Gem .. Qdr Pin Chr ..  Coh HF LLM
```

源码路径：`mem0/utils/factory.py`

| 维度 | 抽象基类 | Provider 数量 | 源码目录 |
|------|---------|:---:|---------|
| LLM | `LLMBase` | 18 | `mem0/llms/` |
| Embedding | `EmbeddingBase` | 11 | `mem0/embeddings/` |
| Vector Store | `VectorStoreBase` | 23 | `mem0/vector_stores/` |
| Reranker | `BaseReranker` | 5 | `mem0/reranker/` |

## 四个抽象基类的接口契约

### LLMBase：推理模型感知的 LLM 抽象

路径：`mem0/llms/base.py`

LLMBase 只定义了一个核心抽象方法，但包含了相当精巧的参数过滤逻辑：

```python
class LLMBase(ABC):
    def __init__(self, config: Optional[Union[BaseLlmConfig, Dict]] = None):
        if config is None:
            self.config = BaseLlmConfig()
        elif isinstance(config, dict):
            self.config = BaseLlmConfig(**config)
        else:
            self.config = config
        self._validate_config()

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto", **kwargs
    ):
        """生成响应。返回 str（无工具）或 dict（有工具调用）。"""
        pass
```

**推理模型检测**是 LLMBase 最独特的设计。OpenAI 的 o1/o3 系列和 GPT-5 系列不支持 `temperature`、`max_tokens` 等参数，强行传入会报错。LLMBase 通过 `_is_reasoning_model()` 方法自动检测模型类型：

```python
def _is_reasoning_model(self, model: str) -> bool:
    # 1. 显式配置优先
    explicit = getattr(self.config, "is_reasoning_model", None)
    if explicit is not None:
        return explicit

    # 2. 名称启发式匹配
    reasoning_models = {
        "o1", "o1-preview", "o3-mini", "o3",
        "gpt-5", "gpt-5o", "gpt-5o-mini", "gpt-5o-micro",
    }
    base_model = model.lower().rsplit("/", 1)[-1]  # 去除 provider 前缀
    if base_model in reasoning_models:
        return True
    # 匹配版本后缀：o1-2024-12-17, o3-2025-04-16
    if any(base_model.startswith(p) for p in ["o1-", "o1.", "o3-", "o3."]):
        return True
    return False
```

当检测到推理模型时，`_get_supported_params()` 会过滤掉不支持的参数，只保留 `messages`、`response_format`、`tools`、`tool_choice` 和可选的 `reasoning_effort`。这种"写一次，所有 Provider 受益"的逻辑正是基类存在的价值。

### EmbeddingBase：支持不对称嵌入

路径：`mem0/embeddings/base.py`

```python
class EmbeddingBase(ABC):
    @abstractmethod
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]]):
        """返回 embedding 向量 (list[float])。"""
        pass

    def embed_batch(self, texts, memory_action="add"):
        """批量嵌入。默认逐条调用 embed()，子类应覆写以获得原生批处理性能。"""
        return [self.embed(text, memory_action) for text in texts]
```

`memory_action` 参数是这个接口最值得关注的设计。某些嵌入模型（如 Google 的 Gecko/Gemini embedding）支持**不对称嵌入**：写入（add）和查询（search）使用不同的 task type，以优化检索质量。通过将动作语义传递到接口层面，Mem0 让每个 Provider 自行决定是否利用这一特性。

`embed_batch()` 提供了带有合理默认实现的模板方法——逐条调用 `embed()`。支持原生批处理的 Provider（如 OpenAI，单次最多 100 条）会覆写此方法以获得显著的性能提升。

### VectorStoreBase：最重的抽象

路径：`mem0/vector_stores/base.py`

VectorStoreBase 定义了 11 个抽象方法和 2 个可选方法，是四个基类中接口最重的：

```python
class VectorStoreBase(ABC):
    # === 必须实现的 11 个方法 ===
    @abstractmethod
    def create_col(self, name, vector_size, distance): ...
    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None): ...
    @abstractmethod
    def search(self, query, vectors, top_k=5, filters=None): ...
    @abstractmethod
    def delete(self, vector_id): ...
    @abstractmethod
    def update(self, vector_id, vector=None, payload=None): ...
    @abstractmethod
    def get(self, vector_id): ...
    @abstractmethod
    def list_cols(self): ...
    @abstractmethod
    def delete_col(self): ...
    @abstractmethod
    def col_info(self): ...
    @abstractmethod
    def list(self, filters=None, top_k=None): ...
    @abstractmethod
    def reset(self): ...

    # === 可选方法（带默认实现）===
    def keyword_search(self, query, top_k=5, filters=None):
        """BM25 关键词搜索。返回 None 表示不支持。"""
        return None

    def search_batch(self, queries, vectors_list, top_k=1, filters=None):
        """批量搜索。默认逐条调用 search()。"""
        return [self.search(q, v, top_k=top_k, filters=filters)
                for q, v in zip(queries, vectors_list)]
```

**分数归一化约束**是 `search()` 方法文档中最关键的一条规则：

> All implementations must return similarity scores where higher values indicate greater similarity (range [0, 1] preferred). Implementations using distance metrics must convert to similarity before returning:
> - Cosine distance: `score = max(0.0, 1.0 - distance)`
> - L2 distance: `score = 1.0 / (1.0 + distance)`
> - Inner product: `score = value` (already higher = better)

这条约束确保了上层的混合评分系统（`mem0/utils/scoring.py`）能够直接将语义分数与 BM25 分数和实体增强分数做加法融合，而不需要知道底层用的是哪个向量数据库。

### BaseReranker：最轻的抽象

路径：`mem0/reranker/base.py`

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]],
               top_k: int = None) -> List[Dict[str, Any]]:
        """对文档按相关性重排序，返回附加 'rerank_score' 字段的文档列表。"""
        pass
```

只有一个方法，契约清晰：接收查询和文档列表，返回重排序后的文档列表，每个文档附加 `rerank_score` 字段。五种实现覆盖了从云 API（Cohere、ZeroEntropy）到本地模型（SentenceTransformer、HuggingFace）再到 LLM-as-a-Judge（LLM Reranker）的全部方案。

## Factory 模式与延迟导入

路径：`mem0/utils/factory.py`

四个工厂类（`LlmFactory`、`EmbedderFactory`、`VectorStoreFactory`、`RerankerFactory`）共享同一个核心机制——`load_class()` 函数：

```python
def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

每个工厂维护一个 `provider_to_class` 字典，将字符串名称映射到完整的模块路径：

```python
class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        "pgvector": "mem0.vector_stores.pgvector.PGVector",
        # ... 共 23 个
    }
```

`create()` 方法在被调用时才通过 `importlib.import_module()` 加载目标模块。这意味着：

1. **按需加载**：如果你只用 Qdrant，就不会加载 Pinecone、Chroma 等其他 22 个 Provider 的依赖
2. **启动零开销**：`import mem0` 不会触发任何 Provider 的导入
3. **优雅降级**：缺少某个 Provider 的依赖包时，只有真正使用它时才会报 `ImportError`

`LlmFactory` 和 `RerankerFactory` 稍微复杂一些，它们的 `provider_to_class` 是元组 `(class_path, config_class)`，因为这两类 Provider 有 Provider 特定的配置类（如 `OpenAIConfig`、`AzureOpenAIConfig`）。工厂在创建实例时会自动处理配置类型的转换。

此外，`LlmFactory` 提供了一个 `register_provider()` 类方法，允许用户在运行时注册自定义 Provider：

```python
@classmethod
def register_provider(cls, name, class_path, config_class=None):
    if config_class is None:
        config_class = BaseLlmConfig
    cls.provider_to_class[name] = (class_path, config_class)
```

## Provider 实现走读

### 案例一：Qdrant 向量存储

路径：`mem0/vector_stores/qdrant.py`

Qdrant 是 Mem0 中实现最完整的 Vector Store Provider，也是唯一原生支持 BM25 混合搜索的。让我们走读关键设计：

**构造函数**支持四种连接方式——注入已有客户端、URL 远程连接、host:port 连接、本地路径存储：

```python
class Qdrant(VectorStoreBase):
    def __init__(self, collection_name, embedding_model_dims,
                 client=None, host=None, port=None,
                 path=None, url=None, api_key=None, on_disk=False):
        if client:
            self.client = client
        else:
            params = {}
            if api_key: params["api_key"] = api_key
            if url: params["url"] = url
            if host and port:
                params["host"] = host; params["port"] = port
            if not params:
                params["path"] = path  # 本地模式
            self.client = QdrantClient(**params)
        # 构造时即创建 collection
        self.create_col(embedding_model_dims, on_disk)
```

**BM25 混合搜索**是 Qdrant Provider 最独特的能力。创建 collection 时会同时配置稠密向量和稀疏向量：

```python
self.client.create_collection(
    collection_name=self.collection_name,
    vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
    sparse_vectors_config={
        "bm25": SparseVectorParams(modifier=models.Modifier.IDF),
    },
)
```

`keyword_search()` 方法使用 fastembed 库的 `Qdrant/bm25` 模型将查询文本编码为稀疏向量，然后对 `bm25` 命名向量槽执行查询。BM25 编码器通过 `_get_bm25_encoder()` 惰性加载，未安装 fastembed 时优雅降级为返回 `None`。

**批量搜索优化**：Qdrant Provider 覆写了 `search_batch()`，使用 Qdrant 的原生 `query_batch_points` API 在单次网络往返中完成多个查询，失败时自动回退到逐条搜索。

### 案例二：OpenAI LLM

路径：`mem0/llms/openai.py`

OpenAI LLM Provider 展示了一个典型的 LLM 实现模式：

```python
class OpenAILLM(LLMBase):
    def __init__(self, config=None):
        # 配置类型转换：dict → OpenAIConfig, BaseLlmConfig → OpenAIConfig
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            config = OpenAIConfig(model=config.model, ...)
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-5-mini"  # 默认模型

        # 支持 OpenRouter 透明代理
        if os.environ.get("OPENROUTER_API_KEY"):
            self.client = OpenAI(api_key=..., base_url=...)
        else:
            self.client = OpenAI(api_key=..., base_url=...)
```

`generate_response()` 的实现流程：

1. 调用基类的 `_get_supported_params()` 获取经过推理模型过滤的参数
2. 合并模型名称、消息列表
3. 条件性添加 `response_format`、`tools`、`tool_choice`、`store` 等参数
4. 调用 `self.client.chat.completions.create(**params)`
5. 通过 `_parse_response()` 统一输出格式：无工具时返回 `str`，有工具时返回 `{"content": ..., "tool_calls": [...]}`
6. 可选调用 `response_callback` 用于响应监控

注意 `store` 参数的处理——只有用户显式配置时才发送给 API，因为 OpenAI 兼容的第三方后端（Gemini、Groq、vLLM）会拒绝未知字段。这种"opt-in 而非 opt-out"的策略是 Provider 实现中的重要模式。

## 添加自定义 Provider

以添加一个假想的 "MyVectorDB" 为例，需要四个步骤：

**步骤 1**：创建实现文件 `mem0/vector_stores/myvectordb.py`：

```python
from mem0.vector_stores.base import VectorStoreBase

class MyVectorDB(VectorStoreBase):
    def __init__(self, collection_name, embedding_model_dims, **kwargs):
        self.client = MyVectorDBClient(**kwargs)
        self.collection_name = collection_name

    def create_col(self, name, vector_size, distance): ...
    def insert(self, vectors, payloads=None, ids=None): ...
    def search(self, query, vectors, top_k=5, filters=None):
        # 关键：分数必须归一化到 [0, 1]
        raw_results = self.client.search(vectors, top_k)
        return [
            Result(id=r.id, score=max(0.0, 1.0 - r.distance), payload=r.metadata)
            for r in raw_results
        ]
    def delete(self, vector_id): ...
    def update(self, vector_id, vector=None, payload=None): ...
    def get(self, vector_id): ...
    def list_cols(self): ...
    def delete_col(self): ...
    def col_info(self): ...
    def list(self, filters=None, top_k=None): ...
    def reset(self): ...
```

**步骤 2**：在 `mem0/configs/vector_stores/` 下添加配置类（如果需要）。

**步骤 3**：注册到 Factory——在 `VectorStoreFactory.provider_to_class` 中添加映射：

```python
"myvectordb": "mem0.vector_stores.myvectordb.MyVectorDB",
```

或者在运行时动态注册（目前仅 `LlmFactory` 提供了 `register_provider()` 方法）。

**步骤 4**：将依赖包添加到 `pyproject.toml` 的可选依赖组（不要加到核心依赖）。

## 设计思考

### 为什么选择 Factory + importlib 而非 Entry Points？

Python 生态有一种更"标准"的插件机制：`setuptools` Entry Points。它允许第三方包通过 `pyproject.toml` 声明插件，无需修改核心代码。那么 Mem0 为什么选择了看似更"原始"的 Factory + importlib 方案？

1. **可发现性**：Factory 的 `provider_to_class` 字典是一个显式的、可读的注册表。开发者打开 `factory.py` 就能看到所有支持的 Provider，而 Entry Points 分散在各个包的 `pyproject.toml` 中。

2. **启动性能**：Entry Points 需要扫描所有已安装包的元数据来发现插件，这在有大量包的环境中可能带来数百毫秒的开销。`importlib.import_module()` 则是精确的按路径加载，零扫描开销。

3. **单体仓库适配**：Mem0 是一个单体仓库，所有 55+ Provider 都在同一个包里。Entry Points 更适合"核心包 + 第三方插件包"的分布式架构。当所有代码都在一个 `mem0ai` 包中时，Factory 模式更自然。

4. **简单性**：Factory 模式不需要理解 `pkg_resources`、`importlib.metadata` 或 `setuptools` 的 Entry Point 规范。任何 Python 开发者都能在 5 分钟内理解它的工作原理。

这个选择反映了 Mem0 的务实哲学：**在当前规模下选择最简单有效的方案**。如果将来 Mem0 需要支持真正的第三方插件生态（让用户 `pip install mem0-provider-myvectordb` 自动注册），Entry Points 会是更好的选择。但在 55 个内置 Provider 的阶段，Factory + importlib 是更优的工程权衡。

### 为什么 VectorStoreBase 要求分数归一化到 [0,1]？

这个约束源于 Mem0 的混合评分架构（详见第 6 章）。搜索路径会将三种信号做加法融合：

```
combined = (semantic_score + bm25_score + entity_boost) / max_possible
```

如果 Qdrant 返回余弦相似度 [0,1]，而 Milvus 返回 L2 距离 [0, +inf]，Pinecone 返回点积 [-1, 1]，这个加法就毫无意义。归一化约束确保所有向量存储的语义分数都在同一尺度上，使得评分公式中的加权系数具有一致的语义。

这是一个**契约式设计**（Design by Contract）的典型案例：基类不强制执行归一化（没有运行时检查），而是通过文档约束将责任下放给实现者。这比在基类中做后处理更灵活——因为某些数据库（如 Qdrant 的余弦搜索）直接返回相似度，不需要额外转换，而在基类中做通用归一化反而会增加不必要的计算。

### keyword_search 为什么是可选的？

并非所有向量数据库都支持全文搜索。FAISS 是纯向量引擎，Pinecone 的 BM25 支持有限。通过将 `keyword_search()` 定义为返回 `None` 的默认方法而非抽象方法，Mem0 实现了一个优雅的渐进增强模式：

- 如果 Provider 支持 BM25（如 Qdrant），覆写方法返回结果，混合评分会融合关键词信号
- 如果 Provider 不支持（如 FAISS），方法返回 `None`，评分系统自动退化为纯语义搜索

这避免了强迫 23 个 Provider 实现一个许多底层引擎根本不支持的功能，同时让支持的 Provider 能够贡献额外的检索信号。

> **下一章**: [05 - 写入路径深度剖析：从对话到记忆的 8 阶段流水线](05-write-path.md)