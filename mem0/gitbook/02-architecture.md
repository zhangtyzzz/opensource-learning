

# 02 - 架构总览：模块结构与数据流

从一个 GitHub 仓库的目录结构开始，我们逐层拆解 Mem0 的架构全貌：从仓库顶层的多语言 SDK 和工具链，到核心 Python SDK 内部的模块分层，再到驱动一切的核心抽象类。本章的目标是让你在脑中建立一张完整的地图，知道每个模块在哪里、做什么、以及它们如何协作。

## 仓库顶层结构：不只是一个 Python 包

Mem0 的 GitHub 仓库远不止一个 Python SDK。它是一个包含多语言客户端、自托管服务、桌面平台和编辑器插件的完整生态：

```
mem0/
  mem0/              # 核心 Python SDK（PyPI: mem0ai）
  mem0-ts/           # TypeScript SDK（npm: mem0ai）
  cli/               # 命令行工具（Python Typer + Node Commander）
  server/            # 自托管 REST 服务（FastAPI + PostgreSQL/pgvector + Neo4j）
  openmemory/        # 自托管记忆平台（FastAPI API + MCP Server + Next.js 15 UI）
  openclaw/          # Claude Code / AI 编辑器插件
  mem0-plugin/       # 编辑器插件（Cursor, Codex 等）
  vercel-ai-sdk/     # Vercel AI SDK 记忆 Provider
  docs/              # Mintlify 文档站
  tests/             # Python SDK 测试（pytest）
  evaluation/        # LOCOMO 基准评测框架
  examples/          # 示例应用、Chrome 扩展、多 Agent 模式
  cookbooks/         # Jupyter Notebooks
```

这些组件的关系可以用一张图来概括：

```
┌─────────────────────────────────────────────────────┐
│                   用户应用层                          │
│  LangChain │ CrewAI │ OpenAI Agents │ 自定义应用      │
└──────┬──────────┬──────────┬───────────────┬────────┘
       │          │          │               │
       ▼          ▼          ▼               ▼
┌─────────────────────────────────────────────────────┐
│                  接入层（多语言 SDK）                  │
│  Python SDK (mem0ai)  │  TypeScript SDK (mem0ai)    │
│  CLI Tools            │  Editor Plugins (OpenClaw)  │
│  Vercel AI Provider   │  MCP Server                 │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
       ▼                          ▼
┌──────────────┐     ┌────────────────────────┐
│   OSS 模式    │     │     Platform 模式       │
│ 本地向量库    │     │   mem0.ai 云端 API      │
│ 本地 SQLite  │     │  （含图记忆、托管存储）    │
│ 本地 LLM 调用│     │                          │
└──────────────┘     └────────────────────────┘
```

关键洞察：**仓库中的大部分组件都是围绕核心 Python SDK 构建的薄层封装**。TypeScript SDK 复刻了 Python SDK 的 OSS 逻辑，CLI 封装了 SDK 的命令行入口，OpenMemory 在 SDK 之上加了 Web UI 和 MCP 协议层。理解了 `mem0/` 这个核心包，就理解了整个生态的 80%。

## Python SDK 模块分层

核心 Python SDK 的内部结构遵循一个清晰的分层模式：

```
mem0/
  __init__.py                  # 入口：导出 Memory, AsyncMemory,
                               #       MemoryClient, AsyncMemoryClient

  memory/                      # 编排层：核心业务逻辑
    base.py                    #   MemoryBase ABC
    main.py                    #   Memory / AsyncMemory（~3280 行）
    storage.py                 #   SQLiteManager（历史 + 消息存储）
    setup.py                   #   配置目录初始化（~/.mem0/）
    telemetry.py               #   PostHog 遥测
    utils.py                   #   消息解析、JSON 提取

  configs/                     # 配置层：Pydantic v2 数据模型
    base.py                    #   MemoryConfig, MemoryItem
    enums.py                   #   MemoryType 枚举
    prompts.py                 #   全部 LLM 提示词（~940 行提取提示词）
    embeddings/base.py         #   BaseEmbedderConfig
    llms/                      #   各 LLM Provider 配置
    vector_stores/             #   各向量库 Provider 配置（22 种）
    rerankers/                 #   各 Reranker 配置

  llms/                        # LLM 适配层（17 种 Provider）
    base.py                    #   LLMBase ABC
    openai.py, anthropic.py, gemini.py, ollama.py, ...

  embeddings/                  # Embedding 适配层（11 种 Provider）
    base.py                    #   EmbeddingBase ABC
    openai.py, huggingface.py, fastembed.py, ...

  vector_stores/               # 向量存储适配层（22 种 Provider）
    base.py                    #   VectorStoreBase ABC
    qdrant.py, pinecone.py, chroma.py, pgvector.py, ...

  reranker/                    # 重排序适配层（5 种 Provider）
    base.py                    #   BaseReranker ABC
    cohere_reranker.py, llm_reranker.py, ...

  client/                      # Platform API 客户端
    main.py                    #   MemoryClient / AsyncMemoryClient
    types.py, project.py, utils.py

  utils/                       # 共享工具层
    factory.py                 #   LlmFactory, EmbedderFactory, ...
    entity_extraction.py       #   spaCy NER（4 种实体类型）
    scoring.py                 #   三信号混合评分
    lemmatization.py           #   BM25 词形还原
```

这个分层的核心设计原则是**关注点分离**：编排层只负责业务流程，不关心具体用哪个 LLM 或向量库；适配层只负责对接外部服务，不关心业务逻辑；配置层将所有可调参数集中管理。

## 核心抽象：六个类撑起整个系统

### MemoryBase：最小接口契约

```python
# mem0/memory/base.py
class MemoryBase(ABC):
    @abstractmethod
    def get(self, memory_id):       # 按 ID 获取单条记忆
        pass
    @abstractmethod
    def get_all(self):              # 获取所有记忆（带过滤）
        pass
    @abstractmethod
    def update(self, memory_id, data):  # 更新记忆内容
        pass
    @abstractmethod
    def delete(self, memory_id):    # 删除记忆
        pass
    @abstractmethod
    def history(self, memory_id):   # 查看记忆变更历史
        pass
```

五个方法，没有 `add()` 和 `search()`——这些被认为是编排层的职责而非接口契约的一部分。这个 ABC 定义的是**数据访问的最小公约数**。

### Memory / AsyncMemory：中央编排器

`mem0/memory/main.py` 是整个系统的心脏，约 3280 行代码。`Memory` 类在初始化时组装所有依赖：

```python
# 简化后的初始化逻辑
class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = None):
        # 1. 解析配置
        self.config = config or MemoryConfig()

        # 2. 通过 Factory 实例化各组件
        self.embedding_model = EmbedderFactory.create(...)
        self.vector_store = VectorStoreFactory.create(...)
        self.llm = LlmFactory.create(...)
        self.reranker = RerankerFactory.create(...)  # 可选

        # 3. 初始化本地存储
        self.db = SQLiteManager(history_db_path)

        # 4. 创建实体存储（独立 collection）
        self.entity_store = VectorStoreFactory.create(
            collection_name=f"{collection}_entities"
        )
```

它承担的职责包括：写入路径的 8 阶段流水线编排、混合检索的三信号融合、实体提取与链接、历史记录管理、消息上下文维护。我们将在第 5 章和第 6 章分别深入这两条路径。

### MemoryConfig：组合式配置

```python
# mem0/configs/base.py（简化）
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig    # 向量库选择与参数
    llm: LlmConfig                    # LLM Provider 与参数
    embedder: EmbedderConfig           # Embedding Provider 与参数
    reranker: Optional[RerankerConfig] # 可选的重排序器
    history_db_path: str               # SQLite 历史库路径
    version: str                       # API 版本（v1.0/v1.1）
    custom_prompt: Optional[str]       # 自定义提取指令
```

Pydantic v2 的嵌套模型让配置的校验和序列化都在初始化时完成，而不是散落在运行时。

### MemoryItem：记忆数据模型

```python
# mem0/configs/base.py（简化）
class MemoryItem(BaseModel):
    id: str               # UUID
    memory: str           # 记忆文本（如 "用户偏好 Python 而非 Java"）
    hash: str             # MD5 哈希（用于去重）
    metadata: dict        # 元数据（user_id, agent_id, 自定义字段）
    score: Optional[float]  # 检索得分（仅 search 结果）
    created_at: datetime
    updated_at: datetime
```

注意 `memory` 字段存储的是**经 LLM 提取的事实陈述**，而非原始对话文本。这是 Mem0 与简单对话日志的本质区别。

### Factory 类：延迟加载的 Provider 解析

```python
# mem0/utils/factory.py（简化）
class LlmFactory:
    provider_to_class = {
        "openai": ("mem0.llms.openai", "OpenAILLM"),
        "anthropic": ("mem0.llms.anthropic", "AnthropicLLM"),
        "ollama": ("mem0.llms.ollama", "OllamaLLM"),
        # ... 17 种 Provider
    }

    @classmethod
    def create(cls, provider_name, config):
        module_path, class_name = cls.provider_to_class[provider_name]
        module = importlib.import_module(module_path)  # 延迟导入
        return getattr(module, class_name)(config)
```

`importlib` 延迟导入意味着：如果你只用 OpenAI，就不会加载 Anthropic、Ollama 等 17 个 Provider 的依赖。这对启动时间和依赖管理至关重要——用户不需要安装所有 Provider 的 SDK。

### SQLiteManager：本地历史存储

```python
# mem0/memory/storage.py（简化）
class SQLiteManager:
    # 两张表：
    # history  — 记忆变更日志（ADD/UPDATE/DELETE + 前后快照）
    # messages — 近期对话消息（每个作用域保留最近 10 条）
```

`history` 表为每条记忆提供完整的审计追踪，`messages` 表为 LLM 提取提供近期对话上下文。这两张表都存在本地 SQLite 文件中（默认 `~/.mem0/history.db`），零外部依赖但也意味着无法直接用于分布式部署。

## 两种运行模式：OSS vs Platform

Mem0 提供两种截然不同的运行模式。用户代码的 API 表面几乎相同，但底层架构完全不同：

```
                    OSS 自托管模式                        Platform 托管模式
                ┌──────────────────┐              ┌──────────────────────┐
                │   你的应用代码    │              │    你的应用代码       │
                │   Memory(config) │              │  MemoryClient(token) │
                └───────┬──────────┘              └──────────┬───────────┘
                        │                                    │
            ┌───────────┼───────────┐                  HTTPS │ API
            │           │           │                        │
            ▼           ▼           ▼                        ▼
      ┌──────────┐ ┌─────────┐ ┌────────┐         ┌─────────────────┐
      │ LLM API  │ │ Vector  │ │SQLite  │         │   mem0.ai 云端   │
      │(OpenAI等)│ │  Store  │ │History │         │                  │
      │          │ │(Qdrant等)│ │  DB   │         │  LLM + Vector +  │
      └──────────┘ └─────────┘ └────────┘         │  Graph(Neo4j) +  │
                                                   │  History + Auth  │
         全部在本地/你的基础设施                       └─────────────────┘
         你管理所有 API Key                            Mem0 托管一切
         无图记忆                                      含图记忆（Pro）
         无 Web UI                                     含管理面板
```

**OSS 模式**（`Memory` 类）：所有组件在本地运行。你需要自己管理 LLM API Key、向量库实例和 SQLite 文件。优势是完全控制、无数据外传（LLM 调用除外）、零额外成本。劣势是没有图记忆、没有 Web 管理界面、不适合分布式部署。

**Platform 模式**（`MemoryClient` 类）：一个纯 HTTP 客户端，所有逻辑在 mem0.ai 云端执行。无需管理任何基础设施。Pro 层（$249/月）提供图记忆（Neo4j）和高级分析。客户端代码极其轻薄——本质上就是带认证的 REST 调用。

两种模式的 API 表面刻意保持一致：

```python
# OSS 模式
from mem0 import Memory
m = Memory.from_config(config_dict)
m.add("我更喜欢用 Python", user_id="alice")
results = m.search("编程语言偏好", user_id="alice")

# Platform 模式
from mem0 import MemoryClient
m = MemoryClient(api_key="xxx")
m.add("我更喜欢用 Python", user_id="alice")
results = m.search("编程语言偏好", user_id="alice")
```

然而，功能上的差距是显著的。以下是 OSS 版本**不包含**的功能：

| 功能 | OSS | Platform |
|------|-----|----------|
| 向量记忆（语义检索） | 有 | 有 |
| 图记忆（实体关系） | **无** | Pro 层 |
| Web 管理面板 | **无**（需自建 OpenMemory） | 有 |
| 多租户 / 团队协作 | **无** | 有 |
| 托管基础设施 | **无** | 有 |
| MCP 服务器 | 需自建 | 托管 |

## 数据流全景：add() 与 search()

在深入细节之前（第 5、6 章），这里先建立写入和读取路径的全局概览。

### 写入路径（add）

```
用户调用 m.add(messages, user_id="alice")
    │
    ▼
Phase 0: 上下文收集
    │  从 SQLite 加载最近 10 条消息
    │  拼接为对话上下文
    ▼
Phase 1: 现有记忆检索
    │  将新消息嵌入为向量
    │  在向量库中检索相似的已有记忆
    ▼
Phase 2: LLM 提取
    │  将对话 + 已有记忆发送给 LLM
    │  LLM 返回 ADD 操作列表（V3 仅 ADD）
    │  每条 ADD 包含: 记忆文本 + linked_memory_ids
    ▼
Phase 3: 批量嵌入
    │  对所有新提取的记忆文本批量生成 embedding
    ▼
Phase 4-5: CPU 处理 + 哈希去重
    │  计算 MD5 哈希
    │  与已有记忆哈希比对，过滤完全重复项
    ▼
Phase 6: 批量持久化
    │  将新记忆写入向量库
    │  将变更记录写入 SQLite history 表
    ▼
Phase 7: 实体链接
    │  用 spaCy 从记忆文本中提取实体
    │  将实体存入独立的 entities collection
    │  建立实体 → 记忆 ID 的链接
    ▼
Phase 8: 消息保存
    │  将原始对话消息存入 SQLite messages 表
    │  保持最近 10 条消息的滑动窗口
    ▼
返回: 新增记忆列表 + 操作结果
```

### 读取路径（search）

```
用户调用 m.search(query, user_id="alice")
    │
    ▼
信号 1: 语义检索
    │  query → embedding → 向量库 top-k 检索
    │  返回 (memory_id, semantic_score) 列表
    │
信号 2: BM25 关键词检索（如向量库支持）
    │  query → 词形还原 → 关键词匹配
    │  返回 (memory_id, bm25_score) 列表
    │
信号 3: 实体增强
    │  query → spaCy 实体提取 → entities collection 检索
    │  找到关联的 memory_id，赋予 entity_boost
    │
    ▼
混合评分
    │  combined = (semantic + sigmoid(bm25) + entity_boost) / max
    │  按 combined 降序排列
    │
    ▼
可选: Reranker 二次排序
    │  将 top-k 结果送入 Reranker（Cohere / LLM / 等）
    │  返回重新排序后的最终结果
    ▼
返回: 排序后的 MemoryItem 列表
```

## 设计思考

### 为什么是单体编排类而非微服务？

`Memory` 类 3280 行代码、承担了从上下文收集到实体链接的全部职责。这在现代软件工程中通常会被视为"上帝类"反模式。但从 Mem0 的定位来看，这个选择是务实的：

**单体的优势**：Mem0 的核心用户场景是"三行代码加记忆"。如果将 8 个 Phase 拆分为独立服务，用户需要部署和配置的组件数量会暴增。单类封装让 `pip install mem0ai` + 实例化 `Memory()` 就能完成全部设置。对于一个开发者工具来说，**入门摩擦的降低**比架构纯度更重要。

**单体的代价**：测试困难（难以独立测试单个 Phase）、代码可读性下降（3280 行需要持续滚动）、扩展受限（想修改 Phase 2 的提取逻辑必须理解整个类）。`AsyncMemory` 与 `Memory` 之间存在大量重复代码，也是这种单体设计的副作用。

值得注意的是，Mem0 确实在**Provider 层面**做了优秀的解耦——55+ 个 Provider 各自独立、通过 Factory 延迟加载。设计者选择了一个折中：**业务流程紧耦合、基础设施松耦合**。这在"核心逻辑稳定、外围集成多变"的场景下是合理的。

### OSS 与 Platform 的功能鸿沟意味着什么？

OSS 版本缺少图记忆是社区最大的不满之一（GitHub issue #4020）。源码中没有 `mem0/graphs/` 目录——图记忆完全是平台独占功能。

这个鸿沟反映的是 Mem0 团队的**商业化策略**：用 OSS 建立开发者心智和社区（51,800 stars），用 Platform 的差异化功能（图记忆、托管、团队协作）驱动付费转化。这与 Elasticsearch（开源搜索 + 商业安全/监控功能）的模式类似。

对使用者的实际影响是：如果你的场景需要实体关系推理（"Alice 的经理是谁？"），OSS 版本的实体链接（基于向量相似度的松散关联）远不如真正的图数据库。你要么付费使用 Platform Pro（$249/月），要么自建 `server/` 组件（需要维护 Docker + Neo4j），要么接受实体链接这个"准图"方案的局限性。

这也解释了为什么 OpenMemory 项目（自托管平台）在社区如此受关注——它试图在 OSS 层面缩小这个差距，尽管目前还没有完整的图记忆支持。

---

本章建立了 Mem0 的全局架构视角。下一章我们将从架构下沉到概念层面，理解记忆类型、身份作用域和记忆生命周期这些核心概念模型。

> **下一章**: [03 - 核心概念：记忆模型与身份作用域](03-core-concepts.md)