# Mem0 深度解读

> 基于源码和官方文档的系统化学习笔记。不是 API 教程，而是理解"为什么这样设计"。

## 目录

| 章节 | 内容 | 状态 |
|------|------|------|
| [01 - Mem0 全景：为 AI 赋予记忆](01-overview.md) | 从 LLM 的无状态困境出发，阐明 Mem0 解决的核心问题：跨会话的个性化记忆。介绍 Mem0 的诞生背景（从 Embedchain 到 Mem0 的演进），核心价值主张（三行代码集成记忆），以及在 AI 记忆生态中的定位。与 Letta/MemGPT、Zep/Graphiti、LangMem、Cognee 等竞品进行架构层面的横向对比，帮助读者判断 Mem0 是否适合自己的场景。设计思考：为什么选择'被动提取'而非'主动记忆'？为什么做 Memory-as-a-Service 而非 Memory-as-a-Library？ | |
| [02 - 架构总览：模块结构与数据流](02-architecture.md) | 从仓库顶层结构（Python SDK、TypeScript SDK、CLI、Server、OpenMemory）到核心 Python SDK 的模块分层，建立全局视角。深入讲解核心抽象（MemoryBase、Memory/AsyncMemory、MemoryConfig、MemoryItem）及其职责边界。用架构图展示两种运行模式——OSS 自托管模式和 Platform 托管模式——的数据流差异。剖析 Memory 类（~3280行）作为中央编排器的角色。设计思考：为什么将 Memory 设计为一个巨型编排类而非拆分为多个微服务？OSS 与 Platform 的功能鸿沟意味着什么？ | |
| [03 - 核心概念：记忆模型与身份作用域](03-core-concepts.md) | 在进入实现细节之前，先建立完整的概念模型。讲解三种记忆类型（情景记忆/语义记忆/程序记忆）的定义和适用场景。深入三级身份作用域（user_id/agent_id/run_id）的隔离机制。阐明记忆生命周期：从 ADD 创建、UPDATE 更新、DELETE 删除到 history 审计追踪的完整链路。解析 MemoryItem 数据模型的每个字段含义，以及 metadata 在过滤和检索中的作用。设计思考：为什么选择三级作用域而非更灵活的标签系统？V3 为什么放弃 UPDATE/DELETE 转向 ADD-only？ | |
| [04 - Provider 插件体系：55+ 实现的统一抽象](04-provider-pattern.md) | 剖析 Mem0 最优雅的架构设计——四维 Provider 插件体系。从四个抽象基类（LLMBase、EmbeddingBase、VectorStoreBase、BaseReranker）出发，讲解每个抽象的接口契约和实现约束。深入 Factory 模式的延迟导入（importlib）策略及其对启动性能的影响。以 Qdrant 向量存储和 OpenAI LLM 为案例，完整走读一个 Provider 的实现过程。讲解如何添加自定义 Provider。设计思考：为什么选择 Factory + importlib 而非 Entry Points？为什么 VectorStoreBase 要求分数归一化到 [0,1]？ | |
| [05 - 写入路径深度剖析：从对话到记忆的 8 阶段流水线](05-write-path.md) | 这是全书技术密度最高的章节。完整剖析 V3 add() 方法的 8 阶段批处理流水线：Phase 0 上下文收集、Phase 1 现有记忆检索、Phase 2 LLM 提取、Phase 3 批量嵌入、Phase 4-5 CPU 处理与哈希去重、Phase 6 批量持久化、Phase 7 实体链接、Phase 8 消息保存。深入解析 940 行提取提示词的设计哲学——为什么同时从用户和助手消息中提取？如何避免'回声提取'？时间引用如何解析？剖析 spaCy 实体提取系统（PROPER/QUOTED/COMPOUND/NOUN 四种类型）和 MD5 哈希去重机制。设计思考：为什么 V3 选择 ADD-only 而非 ADD/UPDATE/DELETE？哈希去重的局限性在哪里？ | |
| [06 - 读取路径深度剖析：三信号混合检索与排序](06-read-path.md) | 完整剖析 search() 方法的混合检索架构。讲解三路信号的产生：语义向量相似度（来自向量存储）、BM25 关键词匹配（经 sigmoid 归一化）、实体增强（实体链接相似度 * 权重）。深入评分公式 combined = (semantic + bm25 + entity_boost) / max_possible 的设计。讲解 Reranker 的二次排序机制（5 种实现：Cohere、HuggingFace、LLM、SentenceTransformer、ZeroEntropy）。分析过滤操作符在元数据层面的查询能力。设计思考：为什么选择加法融合而非学习融合？实体增强权重 0.5 的上限从何而来？ | |
| [07 - 图记忆与平台能力：OSS 之外的世界](07-graph-memory-and-platform.md) | 讲解 Mem0 生态中 OSS SDK 之外的组件。首先剖析图记忆（Graph Memory）的架构：Neo4j 集成、实体-关系建模、Mem0^g 论文中的方法论，以及它为什么在开源版本中缺席（GitHub #4020 社区讨论）。然后介绍自托管 Server 的 Docker Compose 架构（FastAPI + pgvector + Neo4j）。最后讲解 OpenMemory 平台：FastAPI API + MCP 服务器 + Next.js 15 前端的三层架构。设计思考：图记忆付费墙的商业逻辑、自托管 vs 平台托管的取舍矩阵。 | |
| [08 - 集成生态：44+ 框架、MCP 与多语言 SDK](08-integrations.md) | 全面梳理 Mem0 的集成生态。按类型分组讲解：Agent 框架集成（LangChain、CrewAI、AutoGen、OpenAI Agents SDK）、编辑器插件（Claude Code/OpenClaw、Cursor、Codex）、MCP 协议集成、Vercel AI SDK Provider。深入 TypeScript SDK 的架构差异（Client 模式 vs OSS 模式）。用代码示例展示各框架的集成方式。讲解 OpenClaw 插件如何为 Claude Code 提供项目级记忆。设计思考：为什么选择广度优先的集成策略？MCP 协议如何改变记忆层的分发方式？ | |
| [09 - 生产实践：部署、调优与已知局限](09-production-practices.md) | 面向生产环境的实操指南与诚实评估。涵盖部署架构选择（本地 SQLite/Qdrant 单机、Docker Compose 自托管、Platform 托管），性能调优（top_k/threshold 参数、embedding 模型选择、collection 规模规划），多租户模式（单 collection 元数据过滤的利弊）。坦诚分析 6 个已知局限：时间推理弱（LongMemEval 49%）、无隐式模式学习、图记忆付费墙、无时间有效性窗口、embedding 模型锁定、spaCy 依赖开销。用 LOCOMO 和 LongMemEval 基准数据与 Letta、Zep、Evermind 做量化对比。设计思考：Mem0 适合什么场景、不适合什么场景？何时应该选择竞品？ | |

## 对应源码

源码在 `../source/` 目录下（git submodule）。
