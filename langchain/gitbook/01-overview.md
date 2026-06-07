Now I have all the details I need. Let me write the chapter.

# 01 - LangChain 全景：从胶水代码到编排框架

## 一个真实的起点：直接调 SDK 有什么问题？

假设你要构建一个能回答用户问题的客服系统。最直接的方式是调用 OpenAI SDK：

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "我的订单状态是什么？"}]
)
print(response.choices[0].message.content)
```

三行代码，干净利落。但当需求逐步升级时，问题开始暴露：

1. **接数据源** -- 模型需要查询订单数据库，你要写工具调用逻辑、解析函数调用结果、处理重试。
2. **换供应商** -- 老板说试试 Claude，你发现 Anthropic SDK 的消息格式、工具调用协议和 OpenAI 完全不同，得重写一半代码。
3. **加可观测性** -- 生产环境需要追踪每次 LLM 调用的耗时、token 用量、中间推理过程，你要手动埋点。
4. **处理流式输出** -- 用户要求实时看到生成过程，同步/异步/流式三种调用模式各写一遍。
5. **组合多个步骤** -- 先检索文档，再生成回答，再做质量检查，步骤之间的数据流转、错误处理、回退策略全靠手工编排。

这些"胶水代码"不是任何单个问题，而是一类问题：**在 LLM 应用中，模型调用本身只占 20% 的代码量，剩下 80% 是编排、集成、容错和观测**。LangChain 的定位，就是解决这 80%。

## LangChain 是什么

LangChain 是一个 Python（及 JavaScript）开源框架，由 Harrison Chase 于 2022 年底创建，用于构建以大语言模型为核心的应用。其官方定义简洁而准确（来自 `pyproject.toml`）：

> "Building applications with LLMs through composability"

关键词是 **composability（可组合性）**。LangChain 不是一个 LLM wrapper，而是一套让 LLM 调用、数据检索、工具执行、状态管理等组件能够像乐高积木一样拼接的协议和基础设施。

截至写作时（2026 年 6 月），LangChain 在 GitHub 上拥有 117K+ stars，是 LLM 框架领域用户量最大的项目，由风投支持的 LangChain Inc. 维护。最新版本为 `langchain` v1.3.4，核心包 `langchain-core` v1.4.1。

## 三阶段演进：从"魔法封装"到"底层控制"

LangChain 的演进史是理解其设计哲学的最佳入口。这不是一条平滑的增长曲线，而是一次清晰的路线修正。

### 第一阶段（2022-2023）：高层抽象与快速原型

早期 LangChain 的核心卖点是"开箱即用"。用 `LLMChain` 连接 prompt 和模型，用 `RetrievalQA` 一行代码搭 RAG，用 `AgentExecutor` 直接跑 ReAct 循环。对原型开发来说这很爽，但问题很快暴露：

- **抽象过于固化** -- `AgentExecutor` 把推理循环、工具调用、状态管理全部封装在一个类里，想改任何一个环节都要跟整个类搏斗。
- **调试噩梦** -- 一次失败的 Agent 调用，堆栈穿过 5+ 层抽象（Runnable 基类 6,574 行的 `base.py`、CallbackManager、Config 传播），根因定位极其困难。
- **700+ 集成挤在一个包里** -- `pip install langchain` 拉下来一堆你永远不会用的依赖。

### 第二阶段（2024）：模块化拆分与 LangGraph 崛起

这是 LangChain 最重要的路线修正年。几个关键节点：

- **2024 年 2 月**：LangGraph 发布 -- 一个基于图的状态机编排框架，提供比 Chain 更细粒度的控制。
- **2024 年 6 月**：大拆分 -- 700+ 集成从 `langchain` 核心包拆出，变成独立的 partner packages（`langchain-openai`、`langchain-anthropic` 等）。
- **2024 年 10 月**：官方明确宣布 LangGraph 是构建 Agent 的推荐方式，旧式 `AgentExecutor` 开始标记 deprecated。

### 第三阶段（2025 至今）：v1.0 成熟与架构收敛

- **2025 年 10 月**：`langchain` v1.0.0 发布，两个重大变化 --
  1. `langchain` 包将 `langgraph` 作为核心依赖（`pyproject.toml` 第 28 行：`"langgraph>=1.2.4,<1.3.0"`）。
  2. 消息格式更新以支持多模态 Content Block。
- 旧版冻结为 `langchain-classic`（`libs/langchain/`），不再接受新功能，仅做安全修复。
- 新版 `langchain`（`libs/langchain_v1/`）变得极其精简 -- 主要提供 `create_agent` 和 `init_chat_model` 两个工厂函数。

```
演进时间线
──────────────────────────────────────────────────────────────────
2022.10   LangChain 首次发布
          │
2022-2023 第一阶段：LLMChain / AgentExecutor / 单体包
          │  高层抽象，快速原型，但调试困难、依赖臃肿
          │
2024.02   LangGraph 发布（图状态机编排）
2024.06   700+ 集成拆分为独立 partner packages
2024.10   LangGraph 成为官方推荐 Agent 构建方式
          │  第二阶段：模块化 + 底层控制
          │
2025.10   langchain v1.0.0 发布，langgraph 成为核心依赖
          │  第三阶段：架构收敛，langchain-classic 冻结
          │
2026      当前：langchain v1.3.4 / langchain-core v1.4.1
──────────────────────────────────────────────────────────────────
```

## 生态竞品：LangChain 不是唯一选择

LLM 框架领域已经从"LangChain 一家独大"演变为多极竞争。理解竞品的差异化定位，有助于判断 LangChain 的适用边界。

| 框架 | 核心定位 | 与 LangChain 的关系 |
|---|---|---|
| **LlamaIndex** | 数据索引与检索（RAG 专精） | 互补 -- LlamaIndex 管数据，LangChain 管编排；也可独立使用 |
| **CrewAI** | 基于角色的多 Agent 协作 | 竞争 -- 独立于 LangChain，专攻多 Agent 场景 |
| **AutoGen**（Microsoft） | 对话式多 Agent，人机协作 | 竞争 -- 深度集成 Azure，多 Agent 能力更强 |
| **Semantic Kernel**（Microsoft） | 插件化模块，企业级 C#/.NET | 竞争 -- 面向微软技术栈企业客户 |
| **Pydantic AI** | 轻量级、类型安全的 LLM 工具调用 | 竞争 -- 极简主义，适合不需要重框架的场景 |
| **OpenAI Agents SDK** | OpenAI 原生 Agent 框架 | 竞争 -- 仅限 OpenAI 生态，但零抽象开销 |
| **直接 SDK 调用** | 无框架 | "反框架"路线，简单场景下越来越受欢迎 |

**选择决策框架：**

- **简单的单轮 LLM 调用** -> 直接用 OpenAI/Anthropic SDK，无需框架。
- **搜索与检索为核心的 RAG 管线** -> LlamaIndex 更专精，或 LangChain 配合 VectorStore。
- **复杂的多步骤工作流、需要换供应商** -> LangChain + LangGraph 的核心地带。
- **多 Agent 协作场景** -> CrewAI 或 AutoGen 可能更合适。
- **微软技术栈企业** -> Semantic Kernel。
- **类型安全优先、最小依赖** -> Pydantic AI。

LangChain 的核心优势不在任何单一能力，而在**广度**：700+ 集成、最大的社区、最完整的工具链。代价是抽象层的重量。

## LangChain 内部生态：四个项目的分工

LangChain 不是一个单独的库，而是一个产品矩阵：

```
┌─────────────────────────────────────────────────────────┐
│                    LangChain 生态                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  langchain   │  │  LangGraph   │  │  LangSmith   │  │
│  │  (编排+集成) │  │  (Agent状态机)│  │  (追踪+评估) │  │
│  │              │  │              │  │  [商业产品]   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │  依赖            │                  │          │
│         ├─────────────────>│                  │          │
│         │                  │    数据上报       │          │
│         ├──────────────────┼────────────────->│          │
│         │                  │                  │          │
│  ┌──────┴───────┐                                       │
│  │langchain-core│  <── 所有包的公共基础                   │
│  │  (协议+原语) │                                        │
│  └──────────────┘                                       │
│                                                         │
│  ┌──────────────┐                                       │
│  │  LangServe   │  <── API 部署层（REST 化 Runnable）    │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

**langchain-core**：定义所有组件必须遵循的协议 -- Runnable 接口、消息类型、工具规范、回调系统。零重型依赖，是整个生态的地基。

**langchain**（v1）：建立在 `langchain-core` 和 `langgraph` 之上的高层工厂。源码目录 `libs/langchain_v1/langchain/` 实际上非常精简，主要提供：
- `create_agent()` -- 基于 LangGraph 的 Agent 工厂，含 18 个内置中间件
- `init_chat_model()` -- 供应商无关的模型初始化

**LangGraph**：图状态机编排引擎。自 v1.0 起成为 `langchain` 的核心依赖（不再是可选项）。Agent 的推理循环、状态管理、条件分支全部在 LangGraph 层实现。

**LangSmith**：LangChain Inc. 的商业产品。提供追踪（trace）、评估（eval）、监控（monitor）能力。`langchain-core` 的依赖列表中 `langsmith` 是硬依赖（`pyproject.toml` 第 27 行：`"langsmith>=0.3.45,<1.0.0"`），意味着框架级追踪与商业平台存在结构性耦合。

**LangServe**：将任何 Runnable 暴露为 REST API，含自动生成的 playground。定位是部署层，但社区使用率远低于 LangGraph 和 LangSmith。

## 为什么需要读源码

LangChain 的官方文档覆盖了快速入门、集成配置和常见模式（RAG、chatbot、tool use），但有几个关键领域文档严重不足：

1. **Runnable 内部机制** -- 6,574 行的 `base.py` 是整个框架的心脏，但文档对其内部的管道组合、上下文传播、流式拆分的解释远不够深入。
2. **抽象层的调试策略** -- 当 LCEL 管道出错时，如何穿透 5 层抽象定位根因？文档没说。
3. **性能开销的真实量级** -- 社区报告过 40% 的性能差距和 2GB 的内存占用，但官方没有基准测试。
4. **设计决策的 Why** -- 为什么选择管道式组合而非 DAG？为什么回调而非 OpenTelemetry？为什么 v1 要强绑 LangGraph？文档讲 What 和 How，但很少讲 Why。

这本 Gitbook 的目标是填补这些空白：从源码出发，逐层拆解 LangChain 的设计决策、实现细节和工程取舍。

## 设计思考

**为什么一个"LLM 编排框架"需要从追求开箱即用转向提供底层控制力？**

LangChain 第一阶段的设计假设是：LLM 应用的模式是可穷举的 -- RAG、Agent、Chain，每种模式用一个高层类封装就够了。这个假设在 2023 年是合理的，因为大多数人还在做 demo。

但当 LLM 应用进入生产，假设就崩了。原因有三：

1. **生产需求是长尾的**。`AgentExecutor` 封装了一个固定的推理循环（Observation -> Thought -> Action），但真实场景需要条件分支、并行执行、人工审核节点、状态持久化。你需要的不是一个预设好的循环，而是一个能自由组合的状态机。这就是 LangGraph 存在的理由。

2. **抽象会泄漏**。每个 LLM 供应商都有独特能力（OpenAI 的 JSON mode、Claude 的 extended thinking、Gemini 的多模态）。高层抽象为了统一接口必须取最大公约数，这意味着你要么放弃供应商特有功能，要么绕过框架直接调底层 SDK -- 两者都违背了使用框架的初衷。

3. **调试成本与抽象层数成正比**。5 层 Runnable 包装在 happy path 上是透明的，但一旦出错，开发者要理解 RunnableSequence 如何传播 config、CallbackManager 如何分发事件、RunnableBinding 如何覆盖参数 -- 这些都是框架的内部实现，不是应用的业务逻辑。

LangChain v1 的架构选择是对这些教训的直接回应：核心包瘦身到只剩工厂函数，重编排逻辑交给 LangGraph，700+ 集成拆成独立包。本质上，LangChain 从"一个框架做所有事"退化为"一层薄薄的协议 + 一个丰富的生态"。这种退化不是失败，而是成熟 -- 它承认了 LLM 应用的复杂性不能被一层抽象消化，开发者需要的是可组合的原语，而不是预制的解决方案。

但这也带来了新的矛盾：`langchain-core` 对 `langsmith` 的硬依赖意味着"供应商无关"的承诺本身存在一个讽刺性的供应商绑定 -- LangChain 框架与 LangChain 公司的商业产品之间的耦合。这种张力在开源框架商业化的历史中并不罕见（参考 Elastic 与 Elasticsearch、HashiCorp 与 Terraform），但值得使用者在架构选型时纳入考量。

> **下一章**: [02 - 架构与包设计：Monorepo 中的分层解耦](02-architecture.md)