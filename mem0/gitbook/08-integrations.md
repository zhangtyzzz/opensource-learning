

# 08 - 集成生态：44+ 框架、MCP 与多语言 SDK

前七章深入剖析了 Mem0 的内部机制——从写入路径的 8 阶段流水线到读取路径的三信号混合检索。但一个记忆层的价值，最终取决于它能被多少应用触达。本章全面梳理 Mem0 的集成生态：44+ 框架集成、MCP 协议支持、编辑器插件、多语言 SDK，以及支撑这一切的设计策略。

## 集成生态全景

Mem0 的集成版图可以划分为五个层次：

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Applications)                  │
│  Claude Code · Cursor · Codex · 自定义 AI 应用            │
├─────────────────────────────────────────────────────────┤
│               框架集成层 (Framework Integrations)          │
│  LangChain · CrewAI · AutoGen · OpenAI Agents SDK        │
│  Vercel AI SDK · LlamaIndex · Composio · ...             │
├─────────────────────────────────────────────────────────┤
│               协议层 (Protocol Layer)                     │
│  MCP (Model Context Protocol) · OpenAI-Compatible Proxy  │
├─────────────────────────────────────────────────────────┤
│                SDK 层 (Client SDKs)                       │
│  Python SDK (mem0ai) · TypeScript SDK (mem0ai on npm)    │
├─────────────────────────────────────────────────────────┤
│               运行时层 (Runtime)                          │
│  OSS 自托管 · Platform 托管 · OpenMemory (Docker)         │
└─────────────────────────────────────────────────────────┘
```

这种分层架构意味着：一个 CrewAI Agent 通过框架集成层调用 Python SDK，SDK 再与 OSS 本地实例或 Platform API 通信；一个 Claude Code 插件则通过 MCP 协议直接与 OpenMemory 服务器交互。不同层次的集成方式服务于不同的使用场景。

## Agent 框架集成

### LangChain：最成熟的集成

LangChain 是 Mem0 最早支持的框架之一，集成方式体现了 Mem0 "三行代码" 的设计哲学：

```python
from mem0 import MemoryClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

client = MemoryClient(api_key="your-mem0-api-key")
llm = ChatOpenAI(model="gpt-4o")

def chat_with_memory(user_input: str, user_id: str) -> str:
    # 1. 检索相关记忆
    relevant_memories = client.search(user_input, user_id=user_id)
    memories_text = "\n".join(m["memory"] for m in relevant_memories["results"])

    # 2. 构建带记忆上下文的消息
    messages = [
        SystemMessage(content=f"You have these memories about the user:\n{memories_text}"),
        HumanMessage(content=user_input),
    ]
    response = llm.invoke(messages)

    # 3. 将对话存入记忆
    client.add(
        messages=[
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response.content},
        ],
        user_id=user_id,
    )
    return response.content
```

关键设计点：Mem0 不试图成为 LangChain 的 `BaseChatMemory` 子类（LangChain 自己的记忆抽象已经废弃），而是作为一个独立的记忆服务被调用。这避免了与 LangChain 内部 API 变动的耦合。

### CrewAI：多 Agent 场景的记忆共享

CrewAI 的多 Agent 架构使记忆共享变得特别有价值——不同 Agent 需要访问同一用户的偏好和历史：

```python
from crewai import Agent, Task, Crew
from mem0 import MemoryClient

client = MemoryClient(api_key="your-mem0-api-key")

def create_memory_aware_agent(role: str, goal: str, user_id: str) -> Agent:
    # 为 Agent 注入用户记忆作为背景知识
    memories = client.get_all(user_id=user_id)
    memory_context = "\n".join(m["memory"] for m in memories["results"])

    return Agent(
        role=role,
        goal=goal,
        backstory=f"Known facts about the user:\n{memory_context}",
        verbose=True,
    )

# 研究员和写手共享同一用户的记忆
researcher = create_memory_aware_agent(
    "Researcher", "Find relevant information", user_id="alice"
)
writer = create_memory_aware_agent(
    "Writer", "Write personalized content", user_id="alice"
)
```

这里的模式是在 Agent 创建时注入记忆，而非在每次工具调用时检索。这是因为 CrewAI 的 Agent 生命周期较长，背景知识在创建时设定更高效。

### AutoGen：对话式多 Agent 记忆

Microsoft 的 AutoGen 框架强调 Agent 间的对话协作，Mem0 的集成聚焦于跨对话轮次的记忆持久化：

```python
from autogen import ConversableAgent
from mem0 import MemoryClient

client = MemoryClient(api_key="your-mem0-api-key")

def memory_hook(message: dict, sender: str, recipient: str, **kwargs):
    """AutoGen 消息钩子：每轮对话后自动存储记忆"""
    client.add(
        messages=[{"role": "user" if sender == "human" else "assistant",
                   "content": message["content"]}],
        user_id="session-user",
        agent_id=recipient,  # 按 Agent 隔离记忆
    )

agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant with long-term memory.",
    llm_config={"model": "gpt-4o"},
)
```

注意 `agent_id=recipient` 的使用：AutoGen 中不同 Agent 有不同的记忆空间，这正是第 03 章讨论的三级身份作用域的实际应用。

### OpenAI Agents SDK：工具化集成

OpenAI 的 Agents SDK 采用工具（Tool）范式，Mem0 的集成方式是将记忆操作暴露为 Agent 可调用的工具：

```python
from agents import Agent, Tool, Runner
from mem0 import MemoryClient

client = MemoryClient(api_key="your-mem0-api-key")

memory_search_tool = Tool(
    name="search_memory",
    description="Search user's long-term memory for relevant context",
    parameters={"query": {"type": "string"}, "user_id": {"type": "string"}},
    function=lambda query, user_id: client.search(query, user_id=user_id),
)

memory_add_tool = Tool(
    name="save_memory",
    description="Save important information to user's long-term memory",
    parameters={"content": {"type": "string"}, "user_id": {"type": "string"}},
    function=lambda content, user_id: client.add(content, user_id=user_id),
)

agent = Agent(
    name="Memory-Aware Assistant",
    instructions="Use search_memory before responding. Use save_memory for important facts.",
    tools=[memory_search_tool, memory_add_tool],
)
```

这种工具化方式让 Agent 自主决定何时查询和存储记忆，与 Mem0 被动提取模式形成互补：Agent 可以主动保存它认为重要的信息，同时 `add()` 内部的 LLM 提取仍然会从对话中自动提取事实。

## 编辑器插件

### OpenClaw：为 Claude Code 提供项目级记忆

OpenClaw（路径：`openclaw/`）是 Mem0 生态中最有特色的集成之一。它解决的问题是：Claude Code 作为终端 AI 编码助手，每次会话都会遗忘项目上下文。

```
┌─────────────────────────────────────────────┐
│             Claude Code 终端会话              │
│                                              │
│  用户: "重构 auth 模块，用上次讨论的方案"       │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────┐                         │
│  │  OpenClaw 插件   │──── MCP 协议 ────┐      │
│  │  (Claude Code    │                  │      │
│  │   MCP Client)    │                  │      │
│  └─────────────────┘                  │      │
│                                       │      │
│         ▲                             │      │
│         │                             ▼      │
│  ┌──────┴──────┐            ┌──────────────┐ │
│  │  记忆增强的   │            │  OpenMemory  │ │
│  │  响应返回     │◄───────── │  MCP Server  │ │
│  └─────────────┘            └──────────────┘ │
└─────────────────────────────────────────────┘
```

OpenClaw 的核心机制：

1. **项目级记忆隔离**：以项目目录路径作为 `agent_id`，确保不同项目的记忆互不干扰
2. **自动上下文注入**：每次 Claude Code 会话开始时，自动检索与当前项目相关的记忆
3. **对话后自动提取**：会话结束后，自动将有价值的技术决策、架构约定、代码模式存入记忆

安装方式通过 Claude Code 的 MCP 配置完成：

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "uvx",
      "args": ["openmemory-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-key",
        "MEM0_API_KEY": "your-mem0-key"
      }
    }
  }
}
```

### Cursor 与 Codex 集成

Cursor 和 OpenAI Codex 的集成（路径：`mem0-plugin/`）遵循类似模式，但适配各自的插件 API：

- **Cursor**：通过 Cursor 的自定义 AI 规则（Rules）功能注入记忆上下文，配合 MCP 服务器实现记忆的读写
- **Codex**：作为 OpenAI Codex CLI 的记忆后端，在代码生成时提供项目历史上下文

三者的共同设计原则是：记忆层应该是透明的。开发者不需要显式调用记忆 API，插件自动在会话边界处完成记忆的存取。

## MCP 协议集成

Model Context Protocol（MCP）是 Anthropic 提出的开放标准，为 AI 模型提供统一的上下文访问协议。Mem0 通过 `openmemory/` 中的 MCP 服务器实现了这一集成。

### MCP 服务器架构

```
┌────────────────────────────────────────────────────┐
│                OpenMemory MCP Server                │
│                                                     │
│  Tool: add_memory(content, user_id, metadata)       │
│  Tool: search_memory(query, user_id, limit)         │
│  Tool: get_all_memories(user_id)                    │
│  Tool: delete_memory(memory_id)                     │
│  Tool: get_memory_history(memory_id)                │
│                                                     │
│  Transport: stdio (本地) / SSE (远程)                │
│                                                     │
│  Backend: FastAPI + Mem0 Python SDK                  │
└────────────────────────────────────────────────────┘
```

MCP 服务器将 Mem0 的核心 API 暴露为标准 MCP 工具。任何支持 MCP 的客户端——Claude Desktop、Claude Code、Cursor、或自定义 Agent——都可以通过统一协议访问记忆层。

### MCP 改变了什么

在 MCP 之前，每个 AI 应用都需要编写特定的 Mem0 集成代码。MCP 引入了一个关键转变：

```
Before MCP:                        After MCP:
┌──────────┐  custom code          ┌──────────┐
│ App A    │──────────┐            │ App A    │──┐
└──────────┘          │            └──────────┘  │
┌──────────┐          ▼            ┌──────────┐  │  MCP
│ App B    │────► Mem0 SDK         │ App B    │──┼──────► Mem0 MCP
└──────────┘          ▲            └──────────┘  │        Server
┌──────────┐          │            ┌──────────┐  │
│ App C    │──────────┘            │ App C    │──┘
└──────────┘                       └──────────┘

每个应用写自己的集成                  统一协议，一次部署
```

这意味着 Mem0 团队只需维护一个 MCP 服务器实现，而不是为每个 AI 应用分别编写插件。对用户而言，部署一个 OpenMemory MCP 服务器后，所有支持 MCP 的工具都自动获得记忆能力。

## Vercel AI SDK Memory Provider

Vercel AI SDK Provider（路径：`vercel-ai-sdk/`）为 Next.js 和 React 应用提供了服务端记忆集成：

```typescript
import { createMem0 } from "@mem0/vercel-ai-provider";
import { generateText } from "ai";

const mem0 = createMem0({
  apiKey: process.env.MEM0_API_KEY,
  userId: "user-123",
});

// Vercel AI SDK 的 generateText 直接使用 mem0 provider
const { text } = await generateText({
  model: mem0.chatModel("gpt-4o"), // mem0 包装了底层模型
  prompt: "What's my favorite programming language?",
});
```

这个 Provider 的精巧之处在于它包装了底层 LLM 调用：`mem0.chatModel("gpt-4o")` 返回一个兼容 Vercel AI SDK 接口的模型对象，但在内部自动完成记忆检索（请求前）和记忆存储（响应后）。开发者使用标准的 `generateText` / `streamText` API，记忆层完全透明。

## TypeScript SDK 架构

TypeScript SDK（路径：`mem0-ts/src/`）提供了两种截然不同的运行模式：

### Client 模式 vs OSS 模式

```
mem0-ts/src/
  client/       ── MemoryClient：HTTP 客户端，连接 mem0.ai 平台
  oss/          ── Memory：本地运行，自带向量存储和 LLM 调用
  common/       ── 共享类型定义
```

**Client 模式**（`mem0-ts/src/client/`）：

```typescript
import MemoryClient from "mem0ai";

const client = new MemoryClient({ apiKey: "your-api-key" });

// 所有操作通过 HTTP 委托给 mem0.ai 平台
await client.add("I prefer TypeScript over JavaScript", { user_id: "dev-1" });
const results = await client.search("language preference", { user_id: "dev-1" });
```

**OSS 模式**（`mem0-ts/src/oss/`）：

```typescript
import { Memory } from "mem0ai/oss";

const memory = new Memory({
  vectorStore: { provider: "qdrant", config: { url: "http://localhost:6333" } },
  llm: { provider: "openai", config: { model: "gpt-4o" } },
  embedder: { provider: "openai", config: { model: "text-embedding-3-small" } },
});

await memory.add("I prefer TypeScript over JavaScript", { user_id: "dev-1" });
```

### 与 Python SDK 的关键差异

| 维度 | Python SDK | TypeScript SDK |
|------|-----------|---------------|
| Provider 丰富度 | 17 LLM + 11 Embedding + 22 Vector Store | 较少，主流 Provider 为主 |
| 图记忆 | 不支持（OSS 版本） | 不支持 |
| 实体提取 | spaCy NER（4 种实体类型） | 无 spaCy 等效实现 |
| BM25 混合搜索 | 支持（自带 lemmatization） | 依赖向量存储原生能力 |
| 历史审计 | SQLiteManager | 简化实现 |
| 异步支持 | Memory + AsyncMemory 双类 | 原生 async/await |
| 包名 | `mem0ai`（PyPI） | `mem0ai`（npm） |

TypeScript SDK 的 OSS 模式是 Python SDK 的功能子集，这反映了一个务实的优先级判断：大多数 TypeScript 用户运行在 Web 环境中，更可能使用 Platform 托管模式（Client），而非在 Node.js 服务端运行完整的内存管道。

## 集成模式总结

纵观 44+ 集成，可以归纳出三种典型的接入模式：

### 模式一：显式调用（SDK 直接使用）

```python
# 应用代码中直接调用 Mem0 SDK
memories = client.search(query, user_id=uid)
# ... 使用 memories 构建 prompt ...
client.add(messages, user_id=uid)
```

适用于：自定义应用、需要精细控制记忆读写时机的场景。

### 模式二：透明包装（Provider/中间件模式）

```typescript
// Vercel AI SDK Provider：记忆操作对开发者透明
const { text } = await generateText({
  model: mem0.chatModel("gpt-4o"),  // 自动注入记忆
  prompt: userInput,
});
```

适用于：Web 应用、希望最小化集成代码的场景。

### 模式三：协议桥接（MCP/Proxy）

```json
{
  "mcpServers": {
    "openmemory": {
      "command": "uvx",
      "args": ["openmemory-mcp"]
    }
  }
}
```

适用于：AI 编辑器、支持 MCP 的 Agent 框架、需要一次部署服务多个客户端的场景。

## 设计思考

### 为什么选择广度优先的集成策略？

Mem0 目前支持 44+ 集成，覆盖了几乎所有主流 AI 框架和编辑器。这种广度优先策略背后有几个考量：

**网络效应驱动增长**：Memory-as-a-Service 的价值随着接入应用数量增长而增长。每个新集成都降低了下一个开发者的采用成本。51,800 GitHub stars 和 100,000+ 开发者的社区规模，很大程度上归功于 "无论你用什么框架，Mem0 都能接入" 的承诺。

**防御性定位**：AI Agent 框架的竞争格局变化极快——LangChain、CrewAI、AutoGen、OpenAI Agents SDK 各有势力范围。如果 Mem0 只深度绑定一个框架，就会承受该框架衰落的风险。广度集成使 Mem0 成为框架无关的基础设施层。

**代价是什么**：每个集成的深度有限。大多数集成本质上是 "三行代码示例"，没有深度适配框架的原生记忆抽象（比如 LangChain 已废弃的 `BaseChatMemory`）。当框架 API 变动时，44+ 集成的维护负担也不容忽视。

### MCP 协议如何改变记忆层的分发方式？

MCP 的出现代表了一个范式转变：从 "每个应用集成一次 SDK" 到 "部署一个 MCP 服务器，所有应用自动获得记忆能力"。

这对 Mem0 的商业模式有深远影响。在 SDK 集成模式下，Mem0 的价值体现在 Python/TypeScript 包的下载量和 API 调用量。在 MCP 模式下，价值体现在 MCP 服务器的部署数量——一个 MCP 服务器可以同时服务 Claude Code、Cursor、自定义 Agent 等多个客户端，而用户只需在一个地方管理记忆。

MCP 也解释了 OpenMemory 项目的战略意义：它不仅仅是一个自托管方案，更是 Mem0 在 MCP 生态中的关键入口。当 MCP 成为 AI 工具生态的标准协议时，拥有最成熟的记忆 MCP 服务器的项目将获得显著的先发优势。

### TypeScript SDK 的功能差距是有意为之吗？

TypeScript SDK 的 OSS 模式功能远少于 Python SDK——没有 spaCy 实体提取、没有 BM25 混合搜索、Provider 数量更少。这不是资源不足的结果，而是一个有意的产品决策：

TypeScript 用户的典型部署环境是 Vercel、Cloudflare Workers 等 Serverless 平台，这些环境对依赖大小和冷启动时间敏感。完整的混合检索管道（spaCy + BM25 + 实体链接）在这些环境中不切实际。因此，TypeScript SDK 有意引导用户使用 Client 模式连接 Platform，将重计算留在服务端。

这是一个精巧的漏斗设计：TypeScript OSS 模式提供够用的本地体验 -> 用户感受到功能限制 -> 自然升级到 Platform 的全功能服务。

> **下一章**: [09 - 生产实践：部署、调优与已知局限](09-production-practices.md)