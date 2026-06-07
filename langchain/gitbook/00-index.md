# LangChain 深度解读

> 基于源码和官方文档的系统化学习笔记。不是 API 教程，而是理解"为什么这样设计"。

## 目录

| 章节 | 内容 | 状态 |
|------|------|------|
| [01 - LangChain 全景：从胶水代码到编排框架](01-overview.md) | 理解 LangChain 解决的核心问题——为什么直接调用 LLM SDK 在复杂场景下不够用，以及 LangChain 如何在 LLM 框架生态中定位自己。涵盖框架的三阶段演进（高层抽象 -> 模块化拆分 -> v1.0 成熟），与 LlamaIndex、CrewAI、AutoGen、Pydantic AI 等竞品的差异化分析，以及 LangChain 自身生态（LangGraph、LangSmith、LangServe）的协作关系。设计思考：为什么一个'LLM 编排框架'需要从追求开箱即用转向提供底层控制力。 | |
| [02 - 架构与包设计：Monorepo 中的分层解耦](02-architecture.md) | 从代码仓库结构入手，拆解 LangChain 的分层架构：langchain-core（零依赖原语层）、langchain v1（LangGraph 驱动的编排层）、langchain-classic（冻结的遗留层）、partner packages（700+ 集成层）。深入分析依赖流向、包拆分的动机与时机（2024 年 6 月的大拆分）、标准测试套件如何保证集成质量、以及 v1 将 LangGraph 作为核心依赖的重大架构决策。包含 Monorepo 结构图和依赖关系图。设计思考：为什么要把 700+ 集成从核心包中拆出去，以及这对框架治理意味着什么。 | |
| [03 - Runnable 协议：LCEL 的统一计算模型](03-runnable-protocol.md) | 这是 LangChain 最核心的设计——一个 6,574 行的基类如何让所有组件获得统一的 invoke/stream/batch/async 能力。从 Runnable[Input, Output] 泛型接口出发，逐层解析 RunnableSequence（管道 | 操作符）、RunnableParallel（字典并行）、RunnableLambda（函数包装）、RunnableBranch（条件路由）、RunnablePassthrough（透传）的实现原理。深入 RunnableConfig 的上下文传播机制、fallbacks 容错链、retry 重试策略。包含 LCEL 组合流程图。设计思考：为什么选择 Unix 管道式的组合模型而非 DAG 定义，以及这个选择的代价（调试困难、堆栈深度）。 | |
| [04 - 模型与消息：LLM 交互的类型系统](04-models-and-messages.md) | 解析 LangChain 对 LLM 交互的类型化抽象。从 BaseLanguageModel 到 BaseChatModel 的继承层级，理解为什么 Chat 接口成为事实标准。深入消息类型系统：HumanMessage / AIMessage / SystemMessage / ToolMessage 的设计，以及 v1.0 引入的 Content Block 体系如何支持多模态（图片、推理过程、搜索结果）。分析 init_chat_model 工厂函数如何实现供应商无关的模型初始化，以及 ModelProfile 如何描述模型能力元数据。设计思考：为什么 LangChain 选择消息级抽象而非 completion 级抽象，以及 Content Block 设计如何应对多模态演进。 | |
| [05 - Prompt 模板与输出解析：模型 I/O 管线](05-prompts-and-output-parsing.md) | LLM 调用的两端——如何格式化输入（Prompt 模板）和如何结构化输出（Output Parser）。从 BasePromptTemplate 的 Runnable 化设计出发，详解 ChatPromptTemplate 的消息模板组合、变量插值（f-string vs Mustache）、Placeholder 机制。在输出侧，解析 BaseOutputParser 如何将自由文本转化为结构化数据：StrOutputParser、JsonOutputParser、PydanticOutputParser、XMLOutputParser 的适用场景。分析 Few-Shot Prompting 与 ExampleSelector 的设计。设计思考：为什么 Prompt 模板本身也是 Runnable，以及这如何实现模板的可组合性。 | |
| [06 - 工具与函数调用：扩展 LLM 的行动能力](06-tools-and-tool-calling.md) | LLM 从'只能说'到'能做事'的关键抽象。解析 BaseTool 如何将普通函数提升为 LLM 可调用的工具：@tool 装饰器的简洁路径、StructuredTool 的 Pydantic Schema 路径。深入 ToolCall / ToolMessage 协议如何在模型和工具之间传递调用请求与执行结果。分析工具在 LCEL 管道和 Agent 循环中的不同使用模式。设计思考：为什么工具也是 Runnable，以及 ToolCall 协议如何统一不同供应商的函数调用格式。 | |
| [07 - 检索增强生成：从数据加载到语义检索的全链路](07-retrieval-and-rag.md) | LangChain 最常见的应用模式——RAG（检索增强生成）的完整技术栈。从数据进入系统（BaseLoader 加载器）、切分处理（TextSplitter 文本分割）、向量化（Embeddings 接口）、存储索引（VectorStore 向量存储）到查询检索（BaseRetriever 检索器）、文档去重（Indexing API），逐层解析每个抽象的接口设计和典型实现。分析 Document 数据模型为什么刻意区别于消息系统。包含 RAG 数据流图。设计思考：为什么 BaseRetriever 被设计为比 VectorStore 更通用的抽象层，以及 Indexing API 如何解决增量更新问题。 | |
| [08 - Agent 系统：从链式调用到图状态机](08-agents-and-orchestration.md) | LangChain v1 最大的架构变革——Agent 构建从旧式 Chain 全面迁移到 LangGraph 图状态机。解析 create_agent 工厂函数如何在 LangGraph 之上构建 Agent，AgentState TypedDict 如何管理图状态。重点分析中间件（Middleware）系统的设计：human_in_the_loop 人工审核、model_retry / model_fallback 容错、tool_selection 工具选择、pii 脱敏、summarization 上下文压缩、context_editing 上下文编辑等内置中间件的组合模式。以及 SubagentTransformer 子 Agent 委派机制。设计思考：为什么 LangChain 放弃了自己的 Agent 抽象转而依赖 LangGraph，这对框架的定位意味着什么。 | |
| [09 - 可观测性：回调、追踪与事件流](09-observability-and-tracing.md) | LLM 应用的黑盒特性使得可观测性成为生产必需品。解析 LangChain 的双层可观测体系：底层的 Callback 系统（on_llm_start / on_tool_start / on_chain_error 等生命周期钩子）和上层的 Tracer 系统（LangChainTracer 对 LangSmith 的集成、EventStream 事件流、LogStream 日志流）。分析 CallbackManager 如何在 Runnable 执行链中自动传播，以及 astream_events v2 API 如何提供细粒度的执行追踪。设计思考：为什么 LangChain 选择回调而非 OpenTelemetry 等标准方案，以及框架级追踪与商业平台（LangSmith）的耦合问题。 | |
| [10 - 生产实践：安全、性能与框架取舍](10-production-practices.md) | 将 LangChain 从原型推向生产需要面对的现实问题。首先是安全：_security 模块的 SSRF 防护与执行策略、已知 CVE（pickle 反序列化、Prompt 注入、任意代码执行）、2026 年 LangDrained 漏洞事件的教训。然后是性能：框架抽象层的延迟开销（社区报告 40% 性能差距）、内存占用问题（基础检索任务 2GB RAM）、Middleware 层的累积成本。最后是最重要的决策框架：什么时候该用 LangChain、什么时候该直接用 SDK、什么时候该用 LlamaIndex，以及从 langchain-classic 迁移到 v1 的实操路径。设计思考：LangChain 的'供应商无关'承诺是否制造了另一种锁定，以及框架的适用边界在哪里。 | |

## 对应源码

源码在 `../source/` 目录下（git submodule）。
