

# 07 - 图记忆与平台能力：OSS 之外的世界

前六章完整剖析了 Mem0 开源 SDK 的内部机制——从写入路径的 8 阶段流水线到读取路径的三信号混合检索。但如果你仔细对比过 `AGENTS.md` 中提到的"4 Graph Stores: Neo4j, Memgraph, Kuzu, Apache AGE"和实际源码树，你会发现一个显眼的缺口：**`mem0/graphs/` 目录根本不存在**。

这不是 bug，而是有意为之的产品分界线。本章走出 OSS SDK 的边界，探索 Mem0 生态中的三个"外围"组件：图记忆（Graph Memory）的技术架构、自托管 Server 的 Docker Compose 部署、以及 OpenMemory 平台的三层架构。理解这些组件，才能完整把握 Mem0 的产品版图和商业逻辑。

---

## 图记忆：从平面事实到知识图谱

### 为什么需要图记忆

第五章讲过，OSS SDK 将记忆存储为**平面事实**（flat facts）——每条记忆是一个独立的文本片段，通过向量嵌入存入向量数据库。实体链接（Entity Linking）通过一个独立的 `{collection_name}_entities` 向量集合近似实现了关系发现，但本质上仍是"向量相似度匹配"，而非真正的**关系推理**。

考虑这样一个场景：

```
用户说："我的经理 Alice 推荐了 Bob 的论文，Bob 是 Stanford NLP Lab 的博士生。"
```

OSS SDK 会提取出若干独立记忆：

```
- "用户的经理是 Alice"
- "Alice 推荐了 Bob 的论文"
- "Bob 是 Stanford NLP Lab 的博士生"
```

这三条记忆之间没有显式关联。当用户后来问"我经理推荐的那个人在哪个实验室？"时，OSS 的混合检索需要碰巧同时命中多条记忆，再依赖 LLM 的推理能力拼接答案。

图记忆将这些事实建模为**实体-关系三元组**：

```
(User) --[has_manager]--> (Alice)
(Alice) --[recommended_paper_by]--> (Bob)
(Bob) --[affiliated_with]--> (Stanford NLP Lab)
(Bob) --[has_role]--> (PhD Student)
```

现在，"我经理推荐的那个人在哪个实验室？"只需要一条图遍历查询：

```cypher
MATCH (u:User)-[:has_manager]->(mgr)-[:recommended_paper_by]->(person)-[:affiliated_with]->(lab)
RETURN person.name, lab.name
```

### Mem0^g 论文中的方法论

Mem0 团队在其学术论文中提出了 Mem0^g（Mem0 with Graph）架构，核心思路是在标准的向量记忆流水线之上叠加一层知识图谱：

1. **双路提取**：LLM 同时从对话中提取平面事实（存入向量数据库）和实体-关系三元组（存入图数据库）
2. **双路检索**：查询时同时执行向量相似度搜索和图遍历，将两路结果合并后送入 LLM 生成回答
3. **图增强上下文**：图遍历的结果作为额外上下文注入 prompt，增强 LLM 对多跳关系的推理能力

这种架构在需要**关系推理**和**多跳问答**的场景中显著优于纯向量检索。论文报告在 LOCOMO 基准测试中，图增强版本在多跳推理子任务上的提升尤为明显。

### Neo4j 集成架构

在 Mem0 的图记忆实现中，Neo4j 是主要的图数据库后端。架构如下：

```
┌─────────────────────────────────────────────────┐
│                Memory.add()                      │
│                                                  │
│  ┌──────────────┐      ┌──────────────────────┐ │
│  │ Fact Extractor│      │ Relation Extractor   │ │
│  │ (LLM Prompt)  │      │ (LLM Prompt)         │ │
│  └──────┬───────┘      └──────────┬───────────┘ │
│         │                         │              │
│         ▼                         ▼              │
│  ┌──────────────┐      ┌──────────────────────┐ │
│  │ Vector Store  │      │ Neo4j Graph DB       │ │
│  │ (Qdrant/etc.) │      │                      │ │
│  │               │      │ (entity)-[rel]->(entity)│
│  │ flat facts    │      │ Cypher queries       │ │
│  └──────┬───────┘      └──────────┬───────────┘ │
│         │                         │              │
│         └────────┬────────────────┘              │
│                  ▼                               │
│         ┌───────────────┐                        │
│         │ Hybrid Merger  │                        │
│         │ vector + graph │                        │
│         └───────────────┘                        │
└─────────────────────────────────────────────────┘
```

关系提取器使用专门的 LLM prompt，要求模型输出结构化的三元组：

```json
{
  "entities": [
    {"name": "Alice", "type": "PERSON"},
    {"name": "Stanford NLP Lab", "type": "ORGANIZATION"}
  ],
  "relations": [
    {
      "source": "Alice",
      "target": "Stanford NLP Lab",
      "relation": "works_at"
    }
  ]
}
```

这些三元组通过 Neo4j 的 Cypher 查询语言写入图数据库，检索时使用图遍历算法（BFS/DFS）提取相关子图。

### OSS 中缺失的 `mem0/graphs/` 模块

现在来看关键事实：**当前开源仓库中不存在 `mem0/graphs/` 目录**。

`AGENTS.md` 文件明确列出了"4 Graph Stores: Neo4j, Memgraph, Kuzu, Apache AGE"，但这些实现并不在公开的源码树中。GitHub Issue #4020 的社区讨论揭示了这一状况：

- 多位开发者报告无法在 OSS 版本中使用图记忆功能
- 社区成员指出图记忆代码曾短暂出现在某些早期提交中，后被移除
- Mem0 团队回应称图记忆是 Platform Pro 层级的功能

这意味着 `AGENTS.md` 中的描述反映的是**完整代码库**（包括未公开的部分），而非开源发布的子集。开源用户能使用的关系发现能力仅限于第五章讲过的实体链接——基于 spaCy NER 提取实体，存入独立向量集合，在搜索时提供实体增强评分。

---

## 自托管 Server：Docker Compose 架构

对于需要图记忆但不想使用托管平台的用户，Mem0 提供了自托管 Server（位于 `server/` 目录）。这是一个基于 Docker Compose 的完整部署方案。

### 三容器架构

```
┌─────────────────────────────────────────────────────┐
│                 Docker Compose Stack                 │
│                                                     │
│  ┌─────────────────┐  ┌──────────────────────────┐ │
│  │   FastAPI App    │  │   PostgreSQL + pgvector   │ │
│  │                  │  │                          │ │
│  │  REST API        │  │  - Memory vectors        │ │
│  │  /v1/memories/   │  │  - Entity vectors        │ │
│  │  /v1/entities/   │  │  - Metadata filtering    │ │
│  │  /v1/search/     │  │  - BM25 full-text search │ │
│  │                  │  │                          │ │
│  │  Port: 8080      │  │  Port: 5432              │ │
│  └────────┬─────────┘  └──────────────────────────┘ │
│           │                                         │
│  ┌────────┴─────────┐                               │
│  │     Neo4j         │                               │
│  │                   │                               │
│  │  - Entity nodes   │                               │
│  │  - Relationship   │                               │
│  │    edges          │                               │
│  │  - Cypher queries │                               │
│  │                   │                               │
│  │  Port: 7474/7687  │                               │
│  └───────────────────┘                               │
└─────────────────────────────────────────────────────┘
```

**关键组件**：

- **FastAPI App**：暴露与 Platform API 兼容的 REST 接口，处理记忆的 CRUD 操作和搜索请求
- **PostgreSQL + pgvector**：替代 OSS SDK 中可选的 22 种向量数据库，统一使用 pgvector 扩展存储向量嵌入，同时利用 PostgreSQL 原生的全文搜索实现 BM25
- **Neo4j**：提供图记忆能力，存储实体-关系三元组，支持 Cypher 图遍历查询

### 与 OSS SDK 的关键区别

| 特性 | OSS SDK | 自托管 Server |
|------|---------|--------------|
| 向量存储 | 22 种可选后端 | pgvector（固定） |
| 图记忆 | 不可用 | Neo4j（内置） |
| 历史存储 | 本地 SQLite | PostgreSQL |
| API 接口 | Python/TS 直接调用 | REST API（HTTP） |
| 多实例部署 | 不支持（SQLite 锁） | 支持（无状态 API 层） |
| 部署复杂度 | `pip install mem0ai` | Docker Compose + 配置 |

自托管 Server 是介于"轻量 OSS SDK"和"全托管 Platform"之间的中间地带：你获得了图记忆和分布式部署的能力，但需要自行管理 Docker 基础设施、数据库备份、版本升级。

---

## OpenMemory 平台：三层架构

`openmemory/` 目录包含了一个功能更完整的自托管记忆平台，提供 Web UI、MCP 集成和多应用管理能力。

### 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                   OpenMemory Platform                    │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Layer 3: Next.js 15 Frontend           ││
│  │                                                     ││
│  │  - Memory browser (search, filter, delete)          ││
│  │  - Application management dashboard                 ││
│  │  - Memory analytics and statistics                  ││
│  │  - Access control configuration                     ││
│  │                                                     ││
│  │  Port: 3000                                         ││
│  └────────────────────────┬────────────────────────────┘│
│                           │ HTTP                        │
│  ┌────────────────────────┴────────────────────────────┐│
│  │              Layer 2: MCP Server                    ││
│  │                                                     ││
│  │  - Model Context Protocol endpoint                  ││
│  │  - Claude Code / Cursor / AI editor integration     ││
│  │  - Memory read/write via MCP tools                  ││
│  │  - Application-scoped memory isolation              ││
│  │                                                     ││
│  └────────────────────────┬────────────────────────────┘│
│                           │                             │
│  ┌────────────────────────┴────────────────────────────┐│
│  │              Layer 1: FastAPI Backend               ││
│  │                                                     ││
│  │  - REST API for memory CRUD                         ││
│  │  - Qdrant vector store integration                  ││
│  │  - Multi-app memory isolation                       ││
│  │  - User authentication                              ││
│  │                                                     ││
│  │  Port: 8765                                         ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Layer 1: FastAPI Backend

后端 API 层处理所有记忆操作。与自托管 Server 相比，OpenMemory 的后端增加了：

- **多应用隔离**：每个接入的 AI 应用（Claude Code、Cursor 等）有独立的记忆空间
- **访问控制**：基于应用的读写权限管理
- **统计分析**：记忆数量、分布、使用频率等分析数据

### Layer 2: MCP Server

MCP（Model Context Protocol）服务器是 OpenMemory 与 AI 编辑器集成的关键桥梁。它将记忆操作暴露为 MCP 工具，使得 Claude Code、Cursor 等 AI 编辑器可以直接通过标准协议读写记忆：

```json
{
  "tools": [
    {
      "name": "add_memories",
      "description": "Store new memories from the conversation"
    },
    {
      "name": "search_memories",
      "description": "Search for relevant memories"
    },
    {
      "name": "list_memories",
      "description": "List all stored memories"
    },
    {
      "name": "delete_memory",
      "description": "Delete a specific memory"
    }
  ]
}
```

这使得开发者可以在 AI 编码助手中拥有跨会话的项目记忆——代码风格偏好、架构决策、调试历史都可以被记住并在后续会话中自动检索。

### Layer 3: Next.js 15 Frontend

Web 前端基于 Next.js 15 的 App Router 构建，提供：

- **记忆浏览器**：搜索、筛选、查看、删除记忆
- **应用管理**：查看所有接入的 AI 应用及其记忆统计
- **记忆分析**：可视化记忆的创建趋势、分布情况
- **配置管理**：MCP 连接配置、应用权限设置

---

## Platform 定价与功能分层

Mem0 的商业模式通过功能分层实现：

| 功能 | OSS SDK | 自托管 Server | Platform Free | Platform Pro ($249/mo) |
|------|---------|--------------|---------------|----------------------|
| 向量记忆 | 22 种后端 | pgvector | 托管 | 托管 |
| 图记忆 | 不可用 | Neo4j | 不可用 | 内置 |
| 混合检索 | 内置 | 内置 | 内置 | 内置 |
| 实体链接 | spaCy | 增强版 | 增强版 | 增强版 |
| 高级分析 | 不可用 | 基础 | 基础 | 高级 |
| 团队协作 | 不可用 | 不可用 | 有限 | 完整 |
| MCP 集成 | 不可用 | 不可用 | 有限 | 完整 |
| API 限流 | 无 | 自定义 | 有限额 | 高限额 |
| 技术支持 | 社区 | 社区 | 社区 | 优先支持 |

图记忆是 Pro 层级的核心差异化功能。$249/月的定价定位于中型团队和企业级用户——对于个人开发者偏贵，但对于需要生产级记忆基础设施的团队来说，避免了自行运维 Neo4j + pgvector + FastAPI 的工程成本。

---

## 设计思考

### 图记忆付费墙的产品策略

将图记忆限制在付费层级是一个经过深思熟虑的产品决策，而非简单的"功能锁定"：

**商业逻辑**：图记忆是 Mem0 相对于纯向量检索方案（Chroma、Pinecone 原生方案等）的核心技术壁垒。如果完全开源，Mem0 的付费平台将很难与"OSS + 自建基础设施"竞争。图记忆恰好是那种"技术门槛高、维护成本大、但价值显著"的功能——理想的付费功能候选。

**社区张力**：GitHub #4020 的讨论反映了社区的不满。开发者指出 `AGENTS.md` 列出了 4 种图数据库后端，但源码中根本不存在。这种文档与实际的不一致损害了信任。部分开发者选择自行实现简易的图记忆层，或转向 Zep/Graphiti 等原生支持知识图谱的替代方案。

**折中路径**：自托管 Server（`server/` 目录）提供了一条中间路径——通过 Docker Compose 获得图记忆能力，代价是自行管理基础设施。这在一定程度上缓解了社区压力，同时保持了平台的差异化价值。

**评价**：这一策略在商业上合理，但在执行上有改进空间。`AGENTS.md` 应该明确标注哪些模块是开源可用的，哪些是平台独占的。社区期望管理比功能分层本身更重要。

### 自托管 vs 平台托管的决策矩阵

选择哪种部署模式取决于四个维度：

```
                    自行管理基础设施的能力
                    低 ──────────────── 高
                    │                   │
    需要图记忆  是  │  Platform Pro     │  自托管 Server
                    │  ($249/mo)        │  (Docker Compose)
                    │                   │
                否  │  Platform Free    │  OSS SDK
                    │  或 OSS SDK       │  (pip install)
                    │                   │
```

**选择 OSS SDK** 的场景：
- 原型开发、个人项目、学术研究
- 只需要向量记忆，不需要图关系推理
- 已有向量数据库基础设施（如 Qdrant 集群）
- 需要最大程度的灵活性和可定制性

**选择自托管 Server** 的场景：
- 需要图记忆但有数据主权要求（不能将数据发送到外部平台）
- 团队有 Docker/Kubernetes 运维能力
- 对延迟敏感，需要就近部署

**选择 Platform** 的场景：
- 快速上线，不想管理基础设施
- 需要团队协作和高级分析功能
- 愿意为运维成本节约支付月费

**选择 OpenMemory** 的场景：
- 需要 MCP 集成和 Web UI
- 想要多 AI 应用的统一记忆管理
- 介于"纯 SDK"和"全托管平台"之间的需求

### 图记忆的技术取舍

图记忆并非银弹。它引入了额外的复杂度：

1. **双倍 LLM 调用**：需要同时运行事实提取和关系提取两个 prompt，写入延迟和成本翻倍
2. **图数据库运维**：Neo4j 的运维复杂度远高于向量数据库，需要专业的图数据库知识
3. **关系质量依赖 LLM**：如果 LLM 提取的三元组质量不高（关系类型不一致、实体消歧失败），图谱反而会引入噪声
4. **查询复杂度**：Cypher 查询的设计需要领域知识，通用的"从图中找相关信息"比向量搜索更难自动化

对于大多数 AI 应用（聊天助手、客服机器人、个人助理），OSS SDK 的向量记忆 + 实体链接已经足够。图记忆的价值在特定场景下才显现：复杂的多跳问答、组织知识图谱、需要精确关系推理的企业应用。

---

## 小结

Mem0 的产品版图远不止一个 Python SDK。从轻量的 OSS 包到完整的自托管平台，再到托管的 Pro 服务，它覆盖了从个人开发者到企业团队的完整光谱。图记忆是这条光谱上最显眼的分界线——它既是技术上最有价值的能力升级，也是商业上最敏感的付费墙。理解这条分界线的位置和原因，有助于你在项目中做出最合适的技术选型。

> **下一章**: [08 - 集成生态：44+ 框架、MCP 与多语言 SDK](08-integrations.md)