# 10 - 生产实践

## 部署选项

| 方式 | 适用场景 | 说明 |
|------|---------|------|
| 嵌入应用 | 最常见 | 作为 Python 库在你的应用中使用 |
| LangGraph Server | 需要独立部署 | LangGraph Platform 提供的托管服务 |
| Self-hosted Server | 企业内部 | 基于 LangGraph CLI 自行部署 |

## 关键生产配置

### Checkpointer 选择

```python
# ❌ 生产环境不要用
from langgraph.checkpoint.memory import InMemorySaver

# ✅ 生产环境用 PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(conn_string="postgresql://...")
checkpointer.setup()  # 创建表结构
```

### 加密

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# 需要设置环境变量 LANGGRAPH_AES_KEY
serializer = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver(conn_string="...", serde=serializer)
```

### 持久化模式选择

| 模式 | 性能 | 可靠性 | 适用场景 |
|------|------|--------|---------|
| `exit` (默认) | 最好 | 最低 | 短任务、可重试 |
| `async` | 好 | 中 | 大多数场景 |
| `sync` | 一般 | 最高 | 金融、医疗等不可丢失场景 |

## 常见陷阱

### 1. 节点幂等性

**问题**：节点在 interrupt 恢复后会重新执行。

```python
# ❌ 恢复后会重复发邮件
def send_node(state):
    send_email(state["draft"])
    return interrupt("请确认")

# ✅ 用 @task 保证幂等
@task
def send_email_task(draft):
    send_email(draft)

def send_node(state):
    send_email_task(state["draft"]).result()
    return interrupt("请确认")
```

### 2. LastValue 并行写入冲突

**问题**：多个并行节点写同一个 LastValue 字段会报错。

```python
# ❌ 两个并行节点都写 status → 报错
class State(TypedDict):
    status: str  # LastValue channel

# ✅ 用 Reducer 允许并行写入
class State(TypedDict):
    status: Annotated[str, lambda old, new: new]  # 最后一个胜出
```

### 3. Command 和 Edge 冲突

**问题**：一个节点同时有 normal edge 和返回 Command。

```python
# ❌ 冲突
graph.add_edge("A", "B")  # normal edge
def node_A(state):
    return Command(goto="C")  # Command 也指定了跳转

# ✅ 用 conditional edge 或纯 Command，不混用
```

### 4. Checkpoint 存储膨胀

详见 [notes/01-checkpointer-deep-dive.md](../notes/01-checkpointer-deep-dive.md)。

核心策略：
- 使用 `DeltaChannel` 减少冗余存储
- 使用消息裁剪/摘要控制 state 大小
- UI 显示用独立的消息表
- 定期清理旧 checkpoint

### 5. Recursion Limit

```python
# 默认 1000，对于简单 agent 通常足够
# 但复杂 multi-agent 系统可能需要调高
app.invoke(input, {"recursion_limit": 2000})

# 更好的做法：主动检测并降级
class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining: RemainingSteps

def llm_node(state):
    if state["remaining"] < 3:
        return {"messages": [AIMessage("让我总结目前的发现...")]}
    ...
```

## 性能优化

### 减少 LLM 调用

```python
# 节点缓存：相同输入不重复调用
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

graph.add_node("llm", call_llm, cache_policy=CachePolicy(ttl=300))
app = graph.compile(cache=InMemoryCache())
```

### 并行化

利用 LangGraph 的 SuperStep 并行机制：
- 多条出边 → 自动并行
- `Send` → 动态并行
- `@task` → 节点内并行

### 异步执行

```python
# 对于 I/O 密集的节点，使用 async
async def async_node(state):
    result = await async_http_client.get(...)
    return {"data": result}
```

## 可观测性

### LangSmith 集成

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."

# 自动跟踪所有 LLM 调用、节点执行、checkpoint
```

### 自定义 Metadata

```python
app.invoke(input, {
    "metadata": {
        "user_id": "u-123",
        "request_id": "req-456",
    }
})
```

## 图的可视化

```python
# 生成 Mermaid 格式的图
print(app.get_graph().draw_mermaid())

# 生成 PNG（需要 pygraphviz）
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

## Graph Migration（图结构变更）

已完成的 thread 可以兼容大部分图结构变更：

| 变更类型 | 已完成 thread | 被中断的 thread |
|---------|-------------|---------------|
| 添加/删除节点 | 兼容 | 兼容（除非删的是即将执行的节点） |
| 添加/删除 Edge | 兼容 | 兼容 |
| 添加/删除 State 字段 | 兼容 | 兼容 |
| 重命名 State 字段 | 兼容（旧字段数据丢失） | 兼容（旧字段数据丢失） |
| 重命名节点 | 兼容 | 不兼容（如果即将执行该节点） |
