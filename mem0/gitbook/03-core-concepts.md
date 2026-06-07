

# 03 - 核心概念：记忆模型与身份作用域

在深入 Mem0 的实现细节之前，我们需要建立一套完整的概念模型。本章回答三个根本问题：**记什么**（记忆类型）、**给谁记**（身份作用域）、**怎么管**（生命周期与数据模型）。这些概念贯穿后续每一章，是理解写入路径、读取路径和存储层设计的前提。

## 三种记忆类型

Mem0 将记忆分为三种类型，这一分类借鉴自认知心理学中的长时记忆模型，但做了面向工程的简化。

### 情景记忆（Episodic Memory）：对话中的事实

情景记忆记录的是**具体事件和事实**——用户在对话中说了什么、发生了什么。它是 Mem0 最常见的记忆类型，也是 `add()` 方法默认提取的内容。

```python
m.add("我下周三要去上海出差，住在浦东的万豪酒店", user_id="alice")
# 提取出的记忆：
# - "下周三要去上海出差"
# - "出差期间住在浦东的万豪酒店"
```

情景记忆的特点是**时间绑定**且**细节丰富**：提取时会保留专有名词、数量、日期等具体信息。V3 提取提示词（`mem0/configs/prompts.py`）中有明确的指令要求 LLM 保留这些细节而不做概括。

### 语义记忆（Semantic Memory）：用户偏好与知识

语义记忆记录的是**持久性的偏好、特征和知识**——不依附于某次具体对话，而是跨会话有效的信息。

```python
m.add("我是素食主义者，对花生过敏", user_id="alice")
# 提取出的记忆：
# - "是素食主义者"
# - "对花生过敏"
```

语义记忆与情景记忆在 Mem0 的数据模型中**没有显式区分**——它们都是 MemoryItem，存储在同一个向量集合中。区分发生在提取阶段：LLM 根据提示词判断一句话是临时事件还是持久偏好，并据此决定提取的措辞和粒度。

### 程序记忆（Procedural Memory）：指令与规则

程序记忆是三者中最特殊的。它不记录事实，而是记录**行为规则和操作指令**——告诉 AI 助手"应该怎么做"。

```python
from mem0 import Memory

m = Memory()
# 程序记忆通过专门的 API 创建
m.add(
    "当用户询问退款政策时，先确认订单号，再查询订单状态，最后根据7天无理由规则回复",
    user_id="customer_service_agent",
    metadata={"category": "procedural"}
)
```

在源码层面，程序记忆由 `mem0/configs/enums.py` 中的 `MemoryType.PROCEDURAL` 枚举值标识，使用独立的提取提示词（`PROCEDURAL_MEMORY_PROMPT`），并且在 `Memory` 类中有专门的处理分支。程序记忆的核心用途是**塑造 agent 的行为模式**，而非记录用户信息。

### 三种记忆的关系

```
┌─────────────────────────────────────────────────┐
│                   记忆空间                        │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  情景记忆     │  │  语义记忆     │  │ 程序记忆   │ │
│  │             │  │             │  │           │ │
│  │ "周三去上海"  │  │ "素食主义者"  │  │ "先确认    │ │
│  │ "住万豪酒店"  │  │ "花生过敏"   │  │  订单号"   │ │
│  │             │  │             │  │           │ │
│  │ 时间绑定      │  │ 持久有效     │  │ 行为规则   │ │
│  │ 细节丰富      │  │ 跨会话复用   │  │ 指导操作   │ │
│  └─────────────┘  └─────────────┘  └───────────┘ │
│                                                   │
│  存储层：同一个向量集合，通过 metadata 区分          │
└─────────────────────────────────────────────────┘
```

一个关键的设计选择：三种记忆类型在存储层**共享同一个向量集合**。Mem0 没有为每种类型创建独立的 collection。类型区分通过 metadata 字段和提取提示词来实现，检索时统一参与混合搜索。这降低了存储复杂度，但也意味着无法对不同类型的记忆应用不同的检索策略。

## 三级身份作用域

Mem0 用三个 ID 构建记忆的归属关系：`user_id`、`agent_id` 和 `run_id`。这是一个**层级式隔离模型**，每一级都缩小记忆的可见范围。

### user_id：用户级隔离

最常用的作用域。将记忆绑定到特定用户，实现**跨会话的个性化**。

```python
# Alice 的记忆
m.add("我喜欢深色主题", user_id="alice")

# Bob 的记忆
m.add("我喜欢浅色主题", user_id="bob")

# 检索时只返回对应用户的记忆
m.search("主题偏好", user_id="alice")
# -> [{"memory": "喜欢深色主题", ...}]
```

`user_id` 是一个**不透明的字符串标识符**——Mem0 不关心它是数据库主键、邮箱还是 UUID。它作为元数据过滤条件存储在向量数据库中，检索时通过 metadata filter 实现隔离。

### agent_id：Agent 级隔离

将记忆绑定到特定的 AI agent。适用于**多 agent 系统**中不同 agent 需要独立记忆空间的场景。

```python
# 客服 agent 的记忆
m.add("用户偏好中文沟通", user_id="alice", agent_id="customer_service")

# 推荐 agent 的记忆
m.add("用户喜欢科幻类小说", user_id="alice", agent_id="recommendation")

# 客服 agent 只看到自己的记忆
m.search("用户偏好", user_id="alice", agent_id="customer_service")
# -> [{"memory": "用户偏好中文沟通", ...}]
```

`agent_id` 实现的是**同一用户在不同 agent 间的记忆隔离**。客服 agent 不需要知道推荐 agent 记录了什么，反之亦然。

### run_id：会话级隔离

最细粒度的作用域。将记忆绑定到一次特定的运行或会话。适用于需要**会话内记忆但不跨会话持久化**的场景。

```python
# 某次调试会话的上下文
m.add("当前在排查订单 #12345 的支付问题", 
      user_id="alice", run_id="debug_session_001")
```

`run_id` 的典型用途是临时工作上下文——一次调试会话、一个多轮任务流程、或一次特定的交互序列。

### 作用域的组合逻辑

三个 ID 可以自由组合，形成不同粒度的记忆空间：

```
作用域组合                          语义
─────────────────────────────────────────────────
user_id="alice"                    Alice 的所有记忆
agent_id="cs"                     客服 agent 的所有记忆
user_id="alice", agent_id="cs"    Alice 在客服 agent 中的记忆
user_id="alice", run_id="r001"    Alice 在特定会话中的记忆
全部指定                            最精确的记忆空间
全部不指定                          全局记忆（所有用户共享）
```

在 `mem0/memory/main.py` 中，这些 ID 作为 metadata 附加到每条记忆上。关键的过滤逻辑在 `_prepare_filters()` 方法中：

```python
# 简化的过滤逻辑（源码位于 mem0/memory/main.py）
filters = {}
if user_id:
    filters["user_id"] = user_id
if agent_id:
    filters["agent_id"] = agent_id
if run_id:
    filters["run_id"] = run_id
# 这些 filters 传给向量数据库的 metadata filter
```

注意：过滤是**精确匹配**，不是层级包含。指定了 `agent_id="cs"` 就只返回 `agent_id` 恰好等于 `"cs"` 的记忆，不会返回 `agent_id` 为空的记忆。这意味着存入时的作用域组合决定了检索时必须使用的查询范围。

## 记忆生命周期

一条记忆从诞生到消亡，经历一个完整的生命周期。理解这个周期是理解 V2/V3 架构差异的关键。

### V2 模型：ADD / UPDATE / DELETE / NONE

在 V2 版本中（对应 `mem0/configs/prompts.py` 中的 `DEFAULT_UPDATE_MEMORY_PROMPT`），LLM 在提取新记忆后会与现有记忆做对比，为每条新记忆分配一个操作类型：

```
新输入: "我现在改喝燕麦奶了"
现有记忆 #42: "喜欢喝全脂牛奶"

LLM 判断结果:
{
    "id": "42",
    "event": "UPDATE",
    "old_memory": "喜欢喝全脂牛奶",
    "new_memory": "现在改喝燕麦奶（之前喜欢全脂牛奶）"
}
```

四种操作：
- **ADD**：全新的记忆，与现有记忆无冲突
- **UPDATE**：新事实替代或补充已有记忆（就地修改）
- **DELETE**：现有记忆已失效，应当删除
- **NONE**：新输入不包含值得记忆的信息

V2 模型的优势是**记忆空间紧凑**——冲突的事实会被合并而非累积。但问题在于，让 LLM 同时完成"提取新事实"和"判断与旧记忆的关系"是一个**复杂的多任务推理**，容易产生幻觉和错误判断。

### V3 模型：ADD-only + linked_memory_ids

V3（对应 `ADDITIVE_MEMORY_PROMPT`）做了一个大胆的简化：**只做 ADD，不做 UPDATE 和 DELETE**。

```
新输入: "我现在改喝燕麦奶了"
现有记忆 #42: "喜欢喝全脂牛奶"

LLM 判断结果:
{
    "event": "ADD",
    "text": "现在改喝燕麦奶",
    "linked_memory_ids": ["42"]    # 标记与旧记忆的关联
}
```

V3 不再修改或删除旧记忆。取而代之的是通过 `linked_memory_ids` 字段标记新记忆与哪些已有记忆相关。旧记忆仍然存在，但新记忆通过链接表达了"替代"或"补充"关系。

```
V2 生命周期:                       V3 生命周期:
                                   
ADD ──> UPDATE ──> DELETE          ADD ──> ADD ──> ADD ...
  │       │         │                │      │      │
  v       v         v                v      v      v
创建   就地修改    物理删除          创建   追加    追加
                                         (带链接) (带链接)
```

### history 审计追踪

无论 V2 还是 V3，每次记忆操作都会在 SQLite 的 `history` 表中留下审计记录。这由 `mem0/memory/storage.py` 中的 `SQLiteManager` 管理：

```python
# history 表结构（简化）
# memory_id  | prev_value | new_value  | event  | timestamp  | is_deleted
# ---------- | ---------- | ---------- | ------ | ---------- | ----------
# abc-123    | NULL       | "喜欢牛奶"  | ADD    | 2024-01-01 | 0
# abc-123    | "喜欢牛奶"  | "改喝燕麦奶"| UPDATE | 2024-03-15 | 0
# abc-123    | "改喝燕麦奶"| NULL       | DELETE | 2024-06-01 | 1
```

通过 `m.history(memory_id)` 可以查询一条记忆的完整变更记录。这提供了**不可变的审计日志**——即使记忆被更新或删除，历史记录依然保留。

```python
history = m.history("abc-123")
# [
#   {"event": "ADD", "old_memory": None, "new_memory": "喜欢牛奶", ...},
#   {"event": "UPDATE", "old_memory": "喜欢牛奶", "new_memory": "改喝燕麦奶", ...},
#   {"event": "DELETE", "old_memory": "改喝燕麦奶", "new_memory": None, ...}
# ]
```

## MemoryItem 数据模型

每条记忆在 Mem0 内部表示为一个 `MemoryItem`（定义于 `mem0/configs/base.py`）。理解每个字段的含义对调试和扩展至关重要。

```python
class MemoryItem(BaseModel):
    id: str           # UUID v4，记忆的全局唯一标识
    memory: str       # 记忆文本，LLM 提取出的事实陈述
    hash: str         # memory 文本的 MD5 哈希，用于精确去重
    metadata: dict    # 元数据字典，包含作用域 ID 和自定义字段
    score: float      # 检索相关性得分（仅在 search 结果中有值）
    created_at: str   # ISO 8601 创建时间
    updated_at: str   # ISO 8601 最后更新时间
```

### 各字段详解

**id** — 由 `uuid.uuid4()` 生成的唯一标识符。这个 ID 在整个生命周期中不变，用于 `get()`、`update()`、`delete()` 和 `history()` 操作。它也是向量数据库中 payload 的主键。

**memory** — 核心字段。存储的不是原始对话文本，而是 LLM 提取后的**精炼事实陈述**。例如原始输入 "我昨天带我家狗去了宠物医院，它叫小白" 会被提取为 "有一条叫小白的狗" 和 "最近带小白去了宠物医院"。

**hash** — `memory` 字段的 MD5 哈希值。用于写入时的**精确去重**：如果新提取的记忆与已有记忆的 hash 完全相同，则跳过写入。注意这是文本精确匹配——"喜欢喝咖啡" 和 "爱喝咖啡" 会产生不同的 hash，都会被存储。

**metadata** — 一个扁平的字典，承载两类信息：

```python
{
    # 系统元数据（由 Mem0 自动填充）
    "user_id": "alice",
    "agent_id": "customer_service",
    "run_id": "session_001",
    
    # 自定义元数据（由调用者传入）
    "category": "food_preference",
    "confidence": 0.95,
    "source": "onboarding_flow"
}
```

自定义 metadata 的核心价值在于**过滤和检索**。`search()` 和 `get_all()` 都支持基于 metadata 的过滤：

```python
# 只检索食物偏好相关的记忆
m.search("饮食习惯", user_id="alice", filters={"category": "food_preference"})

# 获取所有高置信度的记忆
m.get_all(user_id="alice", filters={"confidence": {"gte": 0.9}})
```

过滤操作符取决于底层向量数据库的支持，但 Mem0 统一抽象了常见的比较操作（`eq`、`ne`、`gt`、`gte`、`lt`、`lte`、`in`、`contains`）。

**score** — 仅在 `search()` 返回的结果中有值，表示该记忆与查询的相关性得分。这个分数由混合检索管线计算（详见第 06 章），融合了语义相似度、BM25 关键词匹配和实体增强三路信号，归一化到 [0, 1] 区间。

**created_at / updated_at** — ISO 8601 格式的时间戳。`created_at` 在记忆首次写入时设置，此后不变。`updated_at` 在每次 UPDATE 操作后刷新。这两个字段存储在向量数据库的 payload 中，也同步到 SQLite 的 history 表。

## 设计思考

### 为什么选择三级作用域而非更灵活的标签系统？

一个显而易见的替代方案是标签系统（tag-based scoping）：给每条记忆打任意数量的标签，检索时用标签组合来过滤。这在理论上更灵活，但 Mem0 选择了固定三级作用域，背后有明确的工程考量。

**可预测性胜过灵活性**。三级作用域的语义是固定的——user 是谁在说、agent 是谁在听、run 是哪次会话。开发者不需要设计标签体系，不会因为标签命名不一致导致记忆泄漏或丢失。在多租户场景下，`user_id` 的隔离语义是确定性的，而标签系统的隔离语义取决于使用者的纪律。

**向量数据库的过滤性能**。大多数向量数据库对固定字段的索引和过滤做了优化（例如 Qdrant 的 indexed payload fields）。三个固定字段可以被预先索引，查询性能可预期。自由标签系统的过滤条件组合爆炸，索引优化困难。

**当然，这也有代价**。如果你需要按 "department"、"project"、"topic" 等维度组织记忆，三级作用域不够用。Mem0 的解决方案是把额外维度放进 `metadata` 字典，通过 `filters` 参数检索。但 metadata 过滤的性能和功能取决于底层向量数据库，不如一等公民的 scope ID 可靠。

### V3 为什么放弃 UPDATE/DELETE 转向 ADD-only？

V2 的 UPDATE/DELETE 模型直觉上更优雅——冲突自动解决，记忆空间保持精简。但在实践中暴露了严重问题。

**LLM 判断的不可靠性**。让 LLM 判断"新事实是否与旧记忆冲突"是一个微妙的推理任务。"我喜欢咖啡" 和 "今天喝了茶" 冲突吗？不一定——喜欢咖啡不意味着不喝茶。但 LLM 可能误判为冲突，执行 UPDATE，丢失了咖啡偏好。这种**误更新造成的信息丢失是不可逆的**。

**提示词复杂度的爆炸**。V2 的提取提示词需要同时指导 LLM 完成四个任务：(1) 从对话中提取事实，(2) 判断新事实与每条现有记忆的关系，(3) 决定操作类型，(4) 生成合并后的文本。这四个任务的交互使得提示词极其复杂，调试困难。

**V3 的 ADD-only 策略**将提取任务简化为单一目标：从对话中提取新事实。冲突检测从"硬删除"降级为"软链接"——通过 `linked_memory_ids` 表达关联，由检索时的排序算法决定哪些记忆更相关。这是一个典型的**写入端简化换取读取端复杂度**的权衡。

代价同样存在。ADD-only 意味着记忆空间会**单调增长**。同一个事实的多个版本都会保留，向量数据库的存储和搜索成本随时间上升。对于长期运行的应用，可能需要定期执行外部清理逻辑。

```
V2: 写入复杂 + 存储紧凑 + 信息丢失风险高
V3: 写入简单 + 存储膨胀 + 信息丢失风险低
```

这个取舍反映了 Mem0 团队的一个核心判断：**在记忆系统中，误删除比冗余存储更危险**。用户可以容忍搜索结果中出现几条过时的记忆，但无法接受重要偏好被错误覆盖后消失。

> **下一章**: [04 - Provider 插件体系：55+ 实现的统一抽象](04-provider-pattern.md)