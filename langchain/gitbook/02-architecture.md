# 02 - 架构与包设计：Monorepo 中的分层解耦

在上一章中，我们理解了 LangChain 要解决的核心问题——为 LLM 应用提供一个供应商无关的编排层。但一个拥有 700+ 集成、横跨多个演进阶段的框架，其内部代码组织不可能是一个简单的 Python 包。本章从仓库结构出发，拆解 LangChain 的分层架构，理解每一层的职责边界、依赖方向，以及推动这种架构形成的工程决策。

## Monorepo 目录结构总览

LangChain 的 Python 实现是一个基于 `uv` 管理的 Monorepo，所有包位于 `libs/` 目录下：

```
langchain/
└── libs/
    ├── core/              # langchain-core v1.4.1 — 原语层
    ├── langchain_v1/      # langchain v1.3.4  — 活跃的编排层
    ├── langchain/         # langchain-classic v1.0.7 — 冻结的遗留层
    ├── partners/          # 第三方集成包（16 个仓库内包）
    │   ├── openai/        #   langchain-openai v1.2.2
    │   ├── anthropic/     #   langchain-anthropic
    │   ├── ollama/        #   langchain-ollama
    │   ├── chroma/        #   langchain-chroma
    │   ├── deepseek/      #   langchain-deepseek
    │   ├── fireworks/     #   langchain-fireworks
    │   ├── groq/          #   langchain-groq
    │   ├── huggingface/   #   langchain-huggingface
    │   ├── mistralai/     #   langchain-mistralai
    │   ├── nomic/         #   langchain-nomic
    │   ├── openrouter/    #   langchain-openrouter
    │   ├── perplexity/    #   langchain-perplexity
    │   ├── qdrant/        #   langchain-qdrant
    │   ├── xai/           #   langchain-xai
    │   └── exa/           #   langchain-exa
    ├── text-splitters/    # langchain-text-splitters v1.1.2
    ├── standard-tests/    # langchain-tests v1.1.9 — 标准测试套件
    └── model-profiles/    # langchain-model-profiles v0.0.5 — 模型元数据 CLI
```

每个子目录都是一个独立版本化的 Python 包，拥有自己的 `pyproject.toml` 和 `uv.lock`。这意味着 `langchain-core` 可以在不发布 `langchain-openai` 新版本的情况下独立升级，反之亦然。

注意：上面列出的只是**主仓库内**的集成包。还有大量集成包（如 `langchain-google-genai`、`langchain-aws`）维护在独立仓库中（如 `langchain-ai/langchain-google`），总数超过 700 个。

## 四层依赖关系

LangChain 的包之间存在严格的单向依赖流：

```
                    ┌──────────────────────┐
                    │   langchain-core     │  ← 零重依赖的原语层
                    │      v1.4.1          │     (pydantic, tenacity,
                    │   ~180 个 .py 文件    │      langsmith, PyYAML)
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                   │
            ▼                  ▼                   ▼
  ┌─────────────────┐ ┌───────────────┐  ┌─────────────────────┐
  │  langchain v1   │ │  langchain-   │  │  partner packages   │
  │    v1.3.4       │ │   classic     │  │  langchain-openai   │
  │  ~32 个 .py 文件 │ │   v1.0.7      │  │  langchain-anthropic│
  │                 │ │  ~1321 个文件   │  │  langchain-ollama   │
  │  + langgraph    │ │  (冻结)        │  │  ...700+ 个         │
  └─────────────────┘ └───────────────┘  └─────────────────────┘
            │                                      │
            └──────────────┬───────────────────────┘
                           │
                    ┌──────▼───────┐
                    │  langchain-  │  ← 辅助包
                    │   tests     │
                    │  text-      │
                    │  splitters  │
                    └─────────────┘
```

关键规则：**依赖箭头永远指向 `langchain-core`，绝不反向**。`langchain-core` 不依赖任何 partner 包，不依赖 `langchain` 本身，不依赖 LangGraph。这是整个架构最重要的不变量。

### 各层的具体依赖声明

从 `pyproject.toml` 中提取的真实依赖关系：

**langchain-core**（原语层）：

```toml
# libs/core/pyproject.toml
dependencies = [
    "langsmith>=0.3.45,<1.0.0",
    "tenacity!=8.4.0,>=8.1.0,<10.0.0",
    "jsonpatch>=1.33.0,<2.0.0",
    "PyYAML>=5.3.0,<7.0.0",
    "typing-extensions>=4.7.0,<5.0.0",
    "packaging>=23.2.0",
    "pydantic>=2.7.4,<3.0.0",
    "uuid-utils>=0.12.0,<1.0",
    "langchain-protocol>=0.0.14",
]
```

**langchain v1**（编排层）——注意极简的依赖列表：

```toml
# libs/langchain_v1/pyproject.toml
dependencies = [
    "langchain-core>=1.4.0,<2.0.0",
    "langgraph>=1.2.4,<1.3.0",        # ← 重大架构决策
    "pydantic>=2.7.4,<3.0.0",
]
```

**langchain-classic**（遗留层）：

```toml
# libs/langchain/pyproject.toml  (package name: langchain-classic)
dependencies = [
    "langchain-core>=1.3.3,<2.0.0",
    "langchain-text-splitters>=1.1.2,<2.0.0",
    "langsmith>=0.1.17,<1.0.0",
    "pydantic>=2.7.4,<3.0.0",
    "SQLAlchemy>=1.4.0,<3.0.0",
    "requests>=2.0.0,<3.0.0",
    "PyYAML>=5.3.0,<7.0.0",
]
```

**partner 包**（以 `langchain-openai` 为例）：

```toml
# libs/partners/openai/pyproject.toml
dependencies = [
    "langchain-core>=1.4.0,<2.0.0",   # ← 只依赖 core
    "openai>=2.26.0,<3.0.0",          # ← 加上供应商 SDK
    "tiktoken>=0.7.0,<1.0.0",
]
```

每个 partner 包的依赖公式都是 `langchain-core + 供应商原生 SDK`。不依赖 `langchain`，不依赖其他 partner 包。

## 三个 "langchain" 包的职责边界

仓库中存在三个容易混淆的包，理解它们的区别至关重要。

### langchain-core：定义"什么是可能的"

**路径**：`libs/core/langchain_core/`（约 180 个 Python 文件）

这是整个框架的地基，包含所有抽象基类和协议定义：

| 模块 | 职责 |
|---|---|
| `runnables/` | Runnable 协议——LCEL 的组合引擎（`base.py` 6,574 行） |
| `language_models/` | `BaseChatModel`、`BaseLanguageModel` 抽象基类 |
| `messages/` | `HumanMessage`、`AIMessage`、`ToolMessage` 等消息类型 |
| `prompts/` | `ChatPromptTemplate`、`PromptTemplate` 模板系统 |
| `output_parsers/` | `BaseOutputParser` 及 JSON/Pydantic/XML 解析器 |
| `tools/` | `BaseTool`、`@tool` 装饰器、`ToolCall` 协议 |
| `vectorstores/` | `VectorStore` 抽象基类 |
| `embeddings/` | `Embeddings` 抽象基类 |
| `retrievers.py` | `BaseRetriever` 抽象基类 |
| `callbacks/` | 回调管理器和生命周期钩子系统 |
| `tracers/` | LangSmith 追踪基础设施 |

`langchain-core` 的设计原则是**零重依赖**。它的依赖列表里没有 `requests`、没有 `SQLAlchemy`、没有任何 LLM 供应商的 SDK。它只定义接口契约，不包含任何具体供应商的实现。

### langchain v1：定义"怎么组合起来"

**路径**：`libs/langchain_v1/langchain/`（约 32 个 Python 文件）

这是当前活跃维护的 `langchain` 包（`pip install langchain` 安装的就是这个）。与庞大的 core 层相比，v1 的代码量极少——它的角色是**高层工厂和编排层**：

| 模块 | 职责 |
|---|---|
| `agents/` | `create_agent` 工厂函数，基于 LangGraph 构建 Agent |
| `agents/middleware/` | 18 个中间件：human-in-the-loop、重试、降级、PII 脱敏等 |
| `chat_models/` | `init_chat_model` 供应商无关的模型初始化工厂 |
| `embeddings/` | Embedding 模型初始化 |
| `tools/` | 工具相关实用函数 |

v1 只有 32 个文件却覆盖了 Agent 创建的全部流程，这是因为真正的计算图引擎来自 LangGraph——`langgraph>=1.2.4,<1.3.0` 是 v1 的核心依赖。

### langchain-classic：定义"过去是什么样的"

**路径**：`libs/langchain/langchain_classic/`（约 1,321 个 Python 文件）

这是 2024 年之前的旧 `langchain` 包，现在以 `langchain-classic` 的名字发布。它包含了大量已废弃的组件：

- 旧式 Chains（`LLMChain`、`ConversationalRetrievalChain` 等）
- 旧式 Agents（`AgentExecutor`、`ZeroShotAgent` 等）
- 旧式 Memory（`ConversationBufferMemory` 等）
- 内置的 document loaders、vectorstores 具体实现

**这个包已被冻结，不接受新功能**。它的存在目的是为旧代码提供向后兼容的迁移过渡期。对比 v1 的 32 个文件和 classic 的 1,321 个文件，可以直观感受到"大拆分"的力度。

## Partner 集成包：标准结构与质量保证

每个 partner 包遵循统一的目录结构。以 `langchain-openai` 为例：

```
libs/partners/openai/
├── langchain_openai/
│   ├── chat_models/       # ChatOpenAI 实现
│   ├── embeddings/        # OpenAIEmbeddings 实现
│   ├── llms/              # 旧式 Completion 接口
│   ├── output_parsers/    # OpenAI 专用解析器
│   ├── tools/             # OpenAI 工具集成
│   ├── middleware/         # 供应商特定中间件
│   └── data/              # Model Profiles（JSON 元数据）
├── tests/
│   ├── unit_tests/        # 单元测试（无网络）
│   └── integration_tests/ # 集成测试（需要 API Key）
├── pyproject.toml
└── uv.lock
```

### standard-tests：用契约测试约束集成质量

`langchain-tests`（`libs/standard-tests/`）是一个共享测试套件，定义了所有集成包必须通过的标准测试。它的设计非常精巧：

**单元测试**（9 个测试方法，`unit_tests/chat_models.py`）：验证初始化、序列化、工具绑定等离线行为。

**集成测试**（48 个测试方法，`integration_tests/chat_models.py`，3,540 行）：验证真实 API 调用，覆盖：

- `test_invoke` / `test_ainvoke` —— 同步和异步调用
- `test_stream` / `test_astream` —— 流式输出
- `test_tool_calling` —— 工具调用
- `test_structured_output` —— 结构化输出
- `test_image_inputs` / `test_pdf_inputs` —— 多模态
- `test_usage_metadata` —— Token 用量追踪
- `test_agent_loop` —— Agent 循环

一个 partner 包要使用这套测试，只需编写极少的粘合代码：

```python
# libs/partners/openai/tests/unit_tests/chat_models/test_base_standard.py
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from langchain_openai import ChatOpenAI

class TestOpenAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"OPENAI_API_KEY": "api_key", "OPENAI_ORG_ID": "org_id", ...},
            {},
            {"openai_api_key": "api_key", "openai_organization": "org_id", ...},
        )
```

继承 `ChatModelUnitTests`，声明你的模型类和初始化参数，标准测试套件自动运行全部 9 个单元测试。

更关键的是 `BaseStandardTests` 中的反作弊机制——`test_no_overrides_DO_NOT_OVERRIDE` 方法会检查子类是否偷偷覆盖了标准测试：

```python
# libs/standard-tests/langchain_tests/base.py
class BaseStandardTests:
    def test_no_overrides_DO_NOT_OVERRIDE(self) -> None:
        # 检测是否有标准测试被删除
        deleted_tests = base_tests - running_tests
        assert not deleted_tests, f"Standard tests deleted: {deleted_tests}"

        # 检测是否有标准测试被覆盖但没有 xfail 标记
        overridden_not_xfail = [
            method for method in overridden_tests if not is_xfail(method)
        ]
        assert not overridden_not_xfail, (
            "Standard tests overridden without @pytest.mark.xfail(reason='...')"
        )
```

如果集成包的某个能力确实无法通过标准测试（比如某个模型不支持工具调用），必须显式标注 `@pytest.mark.xfail(reason="...")` 并给出原因。不允许悄悄删除或覆盖测试。

### Model Profiles：声明式的能力描述

`langchain-model-profiles`（`libs/model-profiles/`）是一个 CLI 工具，从 [models.dev](https://github.com/sst/models.dev) 项目拉取模型元数据，生成供 partner 包使用的 Profile 文件。

每个支持 Profile 的 partner 包在 `data/` 目录下包含：

- `_profiles.py` —— 自动生成的模型能力字典（**不要手动编辑**）
- `profile_augmentations.toml` —— 手动覆盖和补充的配置

例如 `langchain-openai` 的 Profile 记录了每个模型的详细能力：

```python
# libs/partners/openai/langchain_openai/data/_profiles.py（自动生成）
_PROFILES: dict[str, dict[str, Any]] = {
    "chatgpt-image-latest": {
        "name": "chatgpt-image-latest",
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "tool_calling": False,
        "image_outputs": True,
        "reasoning_output": False,
        "max_input_tokens": 0,
        # ...
    },
}
```

当某个模型的实际能力与 models.dev 数据不一致时，`profile_augmentations.toml` 提供覆盖机制：

```toml
# libs/partners/openai/langchain_openai/data/profile_augmentations.toml
provider = "openai"

[overrides]
image_url_inputs = true
pdf_inputs = true
tool_choice = true

[overrides."gpt-3.5-turbo"]
image_url_inputs = false    # gpt-3.5-turbo 不支持图片输入
pdf_inputs = false
```

Profile 数据被标准测试套件用来**自动跳过不适用的测试**——如果一个模型的 Profile 声明 `tool_calling = false`，`test_tool_calling` 就不会对该模型执行。

## v1 对 LangGraph 的核心依赖

`langchain` v1 的 `pyproject.toml` 中只有三个依赖，其中 `langgraph>=1.2.4,<1.3.0` 是最引人注目的一行。这意味着：

1. **Agent 不再是 LangChain 自己实现的**——`create_agent` 工厂函数内部构建的是 LangGraph 状态图
2. **状态管理委托给 LangGraph**——Memory、checkpoint、state persistence 都由 LangGraph 处理
3. **`pip install langchain` 会自动安装 LangGraph**——两个项目在部署上紧密绑定

与之对比，`langchain-classic` 不依赖 LangGraph，它保留了自己的 `AgentExecutor` 实现。这是旧架构和新架构的根本分界线。

v1 的 `agents/middleware/` 目录包含 18 个中间件模块，它们作为可组合的拦截器插入 Agent 的执行流程：

```
middleware/
├── human_in_the_loop.py   # 人工审核关卡
├── model_retry.py         # 模型调用重试
├── model_fallback.py      # 模型降级（如 GPT-4 降级到 GPT-3.5）
├── tool_retry.py          # 工具调用重试
├── tool_selection.py      # 工具选择策略
├── pii.py                 # PII 脱敏
├── summarization.py       # 上下文摘要压缩
├── context_editing.py     # 上下文编辑
├── shell_tool.py          # Shell 命令工具
├── file_search.py         # 文件搜索工具
├── todo.py                # TODO 管理工具
├── tool_call_limit.py     # 工具调用次数限制
├── model_call_limit.py    # 模型调用次数限制
└── tool_emulator.py       # 工具模拟器
```

这些中间件只在 v1 中可用，它们利用 LangGraph 的图节点机制实现拦截——这也解释了为什么 LangGraph 是必须依赖。

## 包拆分的时间线

LangChain 的包结构不是一开始就这样的。理解演进历程有助于理解当前架构的合理性：

**2022-2023**：一个 `langchain` 包包含一切——LLM 抽象、所有集成、Chains、Agents、Memory。安装 `langchain` 会拉入 OpenAI SDK、各种数据库驱动、HTTP 客户端等大量传递依赖。

**2024 年初**：`langchain-core` 被拆分出来，核心抽象独立发版。同期 LangGraph 发布（2024 年 2 月）。

**2024 年 6 月——大拆分**：700+ 集成从 `langchain` 包中拆出，成为独立的 partner 包。原来的 `langchain` 开始瘦身。`langchain-text-splitters` 独立。

**2025 年 10 月——v1.0.0**：旧的 `langchain` 更名为 `langchain-classic` 并冻结。新的 `langchain` v1 以 LangGraph 为核心依赖重新发布。`standard-tests` 和 `model-profiles` 正式成为工具链的一部分。

## 设计思考

### 为什么要把 700+ 集成从核心包中拆出去？

这个决策的驱动力来自三个方面：

**依赖膨胀**。当所有集成都在一个包里时，`pip install langchain` 会通过传递依赖安装大量用户根本不需要的库。一个只用 OpenAI 的项目不应该被迫安装 Chroma、Pinecone 等向量数据库的客户端。Docker 镜像膨胀是社区最频繁的抱怨之一。

**发版耦合**。如果 `langchain-openai` 的一个 bug fix 需要等 `langchain` 整包发版，迭代速度就会被最慢的组件拖累。独立包允许每个集成以自己的节奏发版——OpenAI SDK 更新一个小时后，`langchain-openai` 就可以跟进发版，无需协调其他 700 个集成。

**质量治理**。一个包含 700+ 集成的单体包无法对每个集成施加统一的质量标准。拆分之后，`langchain-tests` 提供了标准化的合规测试套件，每个 partner 包必须通过。`BaseStandardTests` 的反覆盖检测（`test_no_overrides_DO_NOT_OVERRIDE`）确保没有集成能悄悄绕过质量关卡。

### 包拆分对框架治理意味着什么？

这种 Monorepo + 独立发版的架构实际上是一种**联邦治理模型**：

- **核心团队**严格控制 `langchain-core` 的接口稳定性（`_api/deprecation.py` 和 `beta_decorator.py` 提供了 API 版本管理机制）
- **Partner 团队**在标准测试的约束下，对各自的集成包拥有发版自主权
- **接口契约**（抽象基类 + 标准测试）是连接两者的纽带

这也创造了一种有趣的权力动态：`langchain-core` 中任何接口的变更都会波及 700+ 个下游包。这迫使核心团队极其谨慎地对待 breaking changes，同时也解释了为什么 `CLAUDE.md` 中用**加粗大写**写着"CRITICAL: Always attempt to preserve function signatures"。

### LangGraph 作为核心依赖的风险与回报

将 LangGraph 设为 `langchain` v1 的硬依赖是一个大胆的决策。回报是显而易见的：Agent 系统获得了图状态机的全部能力（条件分支、循环、持久化、人工审核节点），而 LangChain 团队不需要自己维护一套状态管理系统。

但这也创造了紧耦合：LangGraph 的 API 变更会直接影响 `langchain` v1 的稳定性。版本约束 `langgraph>=1.2.4,<1.3.0` 表明 LangChain 团队对此是清醒的——他们锁定了 LangGraph 的次版本号，以限制意外 breaking change 的风险。

从更宏观的角度看，这个决策标志着 LangChain 从"全栈 LLM 框架"向"LLM 原语层 + 编排层胶水"的角色转变。核心计算模型（Runnable/LCEL）仍然在 `langchain-core` 中，但复杂的有状态编排已经外包给了 LangGraph。这是一种务实的收缩——与其在一个包中做所有事情（然后做不好），不如专注于做好接口定义和组合粘合，把状态机引擎的工作交给专门的项目。

> **下一章**: [03 - Runnable 协议：LCEL 的统一计算模型](03-runnable-protocol.md)