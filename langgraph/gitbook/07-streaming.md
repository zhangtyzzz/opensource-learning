# 07 - Streaming 机制

## 为什么 Streaming 重要

Agent 的一次执行可能持续数十秒甚至数分钟（多轮 LLM 调用 + 工具执行）。如果用户要等到全部完成才看到结果，体验极差。

Streaming 让用户在 Agent **执行过程中**就能看到：
- LLM 正在生成什么（token 级）
- 当前在执行哪个节点
- 工具调用的结果
- 自定义的进度信息

## 7 种 Stream 模式

| 模式 | 输出内容 | 典型用途 |
|------|---------|---------|
| `values` | 每步后的**完整** state | 展示全局状态变化 |
| `updates` | 每步的 state **增量** | 展示"这一步改了什么" |
| `messages` | LLM 的 **token 流** | 打字机效果 |
| `custom` | 开发者自定义数据 | 进度条、中间结果 |
| `checkpoints` | Checkpoint 事件 | 调试、状态同步 |
| `tasks` | 任务开始/完成事件 | 进度跟踪 |
| `debug` | 以上全部 | 开发调试 |

### 基本用法（v2 格式）

```python
# 单一模式
for chunk in app.stream(input, config, stream_mode="updates"):
    print(chunk)
    # {"type": "updates", "ns": (), "data": {"llm": {"messages": [...]}}}

# 多模式组合
for chunk in app.stream(input, config, stream_mode=["updates", "messages"]):
    if chunk["type"] == "updates":
        handle_state_update(chunk["data"])
    elif chunk["type"] == "messages":
        handle_token(chunk["data"])
```

v2 格式的每个 chunk 都是一个 dict：
- `type` — 哪种模式产生的
- `ns` — 命名空间（子图时有值）
- `data` — 实际数据

### Token 级流式 (`messages`)

```python
for chunk in app.stream(input, config, stream_mode="messages"):
    msg_chunk, metadata = chunk["data"]
    # msg_chunk: AIMessageChunk (单个 token 或几个 token)
    # metadata: {"langgraph_node": "llm", "tags": [...]}

    print(msg_chunk.content, end="", flush=True)  # 打字机效果
```

**过滤技巧**：

```python
# 按节点过滤（只看 "llm" 节点的输出）
if metadata["langgraph_node"] == "llm":
    print(msg_chunk.content, end="")

# 按 tag 过滤
llm = ChatOpenAI(model="gpt-4o").with_config(tags=["primary"])
# 然后在 stream 循环中检查 metadata["tags"]

# 静默某个 LLM（不流式输出）
silent_llm = ChatOpenAI(model="gpt-4o-mini").with_config(tags=["nostream"])
```

### 自定义流式 (`custom`)

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()

    writer({"progress": 0.3, "status": "正在搜索..."})
    results = search(state["query"])

    writer({"progress": 0.7, "status": "正在分析..."})
    analysis = analyze(results)

    writer({"progress": 1.0, "status": "完成"})
    return {"analysis": analysis}
```

也可以从 **Tool 内部** 发送自定义流：

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str, config: RunnableConfig) -> str:
    writer = get_stream_writer()
    writer({"tool_status": f"搜索: {query}"})
    return do_search(query)
```

## 架构：Transformer 管道

```python
# source/libs/langgraph/langgraph/stream/
```

流式输出的内部架构是一个**转换器管道**：

```
Pregel 引擎
  │  产生原始执行事件
  ▼
GraphRunStream (stream/run_stream.py)
  │  分发给各 transformer
  ▼
┌─────────────────────────────────┐
│ UpdatesTransformer   → updates  │
│ CheckpointsTransformer → checkpoints │
│ DebugTransformer     → debug    │
│ CustomTransformer    → custom   │
│ TasksTransformer     → tasks    │
│ LifecycleTransformer → lifecycle │
│ SubgraphTransformer  → subgraph │
└─────────────────────────────────┘
  │  各自投影为不同视角
  ▼
用户的 stream 迭代器
```

每个 Transformer 只关注自己的事件类型，过滤掉其他事件。`stream_mode` 参数决定哪些 Transformer 被激活。

### 子图 Streaming

当图中包含子图时，`subgraphs=True` 启用子图事件：

```python
for chunk in app.stream(input, config, subgraphs=True):
    print(chunk["ns"])  # () = 主图, ("inner:uuid",) = 子图
```

`ns` 字段是命名空间元组，表示事件来自哪个嵌套层级。

## v1 vs v2 格式

| 维度 | v1 (< 1.1) | v2 (>= 1.1) |
|------|-----------|-------------|
| 格式 | 裸 tuple / dict | 统一 `{"type", "ns", "data"}` |
| 多模式区分 | 靠位置 | 靠 `type` 字段 |
| 子图命名空间 | 无 | `ns` 元组 |
| 类型安全 | 弱 | 可以 type narrowing |

**建议**：新项目直接用 v2。

## 与 Interrupt 的交互

当图在 streaming 过程中遇到 `interrupt()`：

1. 当前 chunk 正常发出
2. 图暂停，创建 checkpoint
3. **stream 不会中断**——它会正常结束迭代
4. 最终的 `invoke` 返回值中包含 `.interrupts` 信息

```python
result = app.invoke(input, config)
if result.interrupts:
    # 图被中断了，需要人工处理
    print(result.interrupts)
```

> **下一章**: [08 - Human-in-the-Loop](08-human-in-the-loop.md) — 在 Agent 执行中引入人工决策
