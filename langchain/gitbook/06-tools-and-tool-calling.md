# 06 - 工具与函数调用：扩展 LLM 的行动能力

LLM 本身只能生成文本。要让它查数据库、调 API、执行计算，需要一个桥梁——**工具（Tool）**。LangChain 的工具抽象解决两个核心问题：如何把普通 Python 函数"提升"为 LLM 可理解和调用的工具，以及如何在模型与工具之间建立标准化的请求-响应协议。

本章基于 `libs/core/langchain_core/tools/` 目录的源码展开分析。

## BaseTool：继承层级与 Runnable 化

工具体系的根是 `BaseTool`，定义在 `langchain_core/tools/base.py`。它的类型签名揭示了一个关键设计决策：

```python
class BaseTool(RunnableSerializable[str | dict | ToolCall, Any]):
    name: str
    description: str
    args_schema: ArgsSchema | None = None
    return_direct: bool = False
    response_format: Literal["content", "content_and_artifact"] = "content"
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = False
```

`BaseTool` 继承自 `RunnableSerializable`，这意味着每个工具天然就是一个 Runnable。它的输入类型是 `str | dict | ToolCall` 的联合类型——工具既能接受简单字符串、字典参数，也能接受标准化的 `ToolCall` 对象。这三种输入在 `invoke` 时由 `_prep_run_args` 统一处理：

```python
# langchain_core/tools/base.py
def _prep_run_args(value: str | dict | ToolCall, config, **kwargs):
    if _is_tool_call(value):  # 检查 type == "tool_call"
        tool_call_id = cast(ToolCall, value)["id"]
        tool_input = cast(ToolCall, value)["args"].copy()
    else:
        tool_call_id = None
        tool_input = cast("str | dict", value)
    return tool_input, dict(callbacks=..., tool_call_id=tool_call_id, ...)
```

当输入是 `ToolCall` 时，框架自动提取 `args` 和 `id`；当输入是原始字符串或字典时，`tool_call_id` 为 `None`。这个分支决定了输出格式——有 `tool_call_id` 时返回 `ToolMessage`，没有时返回原始内容。

工具的完整继承层级：

```
RunnableSerializable[str | dict | ToolCall, Any]
    └── BaseTool              (抽象基类，定义 _run/_arun)
         ├── Tool             (单输入简单工具)
         └── StructuredTool   (多参数结构化工具，@tool 装饰器的产物)
```

## @tool 装饰器：最简路径

`@tool` 装饰器定义在 `langchain_core/tools/convert.py`，是创建工具最常用的方式：

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网上的信息。"""
    return f"Results for: {query}"

# search 现在是一个 StructuredTool 实例
print(search.name)         # "search"
print(search.description)  # "搜索互联网上的信息。"
print(search.args)         # {"query": {"type": "string"}}
```

装饰器内部的执行流程：

1. 从函数的 `__name__` 获取工具名称
2. 从 docstring 获取描述
3. 调用 `create_schema_from_function` 从类型注解生成 Pydantic schema
4. 构造并返回一个 `StructuredTool` 实例

`@tool` 支持多种用法：

```python
# 1. 无参数装饰器——名称取自函数名
@tool
def my_func(x: str) -> str: ...

# 2. 自定义名称
@tool("custom_name")
def my_func(x: str) -> str: ...

# 3. 解析 Google 风格 docstring 为参数描述
@tool(parse_docstring=True)
def calculate(expression: str, precision: int = 2) -> float:
    """计算数学表达式。

    Args:
        expression: 要计算的数学表达式。
        precision: 小数精度。
    """
    ...

# 4. 返回 content + artifact
@tool(response_format="content_and_artifact")
def search(query: str) -> tuple[str, list[dict]]:
    """搜索文档。"""
    results = [{"title": "doc1", "score": 0.95}]
    return "Found 1 result", results
```

`parse_docstring=True` 值得注意——它将 Google 风格 docstring 的 `Args` 部分解析为 JSON Schema 的 `description` 字段，让 LLM 更好地理解每个参数的含义。

## StructuredTool 与 Pydantic Schema

当需要更精细地控制输入 schema 时，可以直接使用 `StructuredTool`（`langchain_core/tools/structured.py`）：

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=10, description="最大返回数量")
    language: str = Field(default="zh", description="结果语言")

def search_impl(query: str, max_results: int = 10, language: str = "zh") -> str:
    return f"Searching '{query}' (max={max_results}, lang={language})"

search_tool = StructuredTool.from_function(
    func=search_impl,
    name="web_search",
    description="搜索互联网",
    args_schema=SearchInput,
)
```

Pydantic schema 的价值在于：它会被序列化为 JSON Schema，然后通过 `bind_tools` 发送给 LLM。LLM 根据这个 schema 生成结构化的调用参数，而框架再用同一个 Pydantic 模型验证 LLM 的输出——形成一个闭环。

`BaseTool` 的 `tool_call_schema` 属性在导出给 LLM 的 schema 中会自动过滤掉标记为 `InjectedToolArg` 的参数。这些注入参数（如 `RunnableConfig`、`tool_call_id`）是框架运行时注入的，不应暴露给模型：

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

@tool
def greet(
    name: str,
    user_id: Annotated[str, InjectedToolArg],  # LLM 看不到这个参数
) -> str:
    """问候用户。"""
    return f"Hello {name} (uid={user_id})"
```

## ToolCall / ToolMessage 协议

`ToolCall` 和 `ToolMessage` 是工具调用的请求-响应协议，定义在 `langchain_core/messages/tool.py`。

**ToolCall** 是一个 TypedDict，表示模型发出的调用请求：

```python
class ToolCall(TypedDict):
    name: str                              # 工具名称
    args: dict[str, Any]                   # 调用参数
    id: str | None                         # 调用 ID（关联请求和响应）
    type: NotRequired[Literal["tool_call"]] # 类型标识
```

它存在于 `AIMessage.tool_calls` 列表中——当模型决定调用工具时，返回的 `AIMessage` 会包含一个或多个 `ToolCall`：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([search_tool])

response = llm_with_tools.invoke("帮我搜索 LangChain 最新版本")
# response.tool_calls 可能是:
# [{"name": "web_search", "args": {"query": "LangChain latest version"}, "id": "call_abc123"}]
```

**ToolMessage** 是 BaseMessage 的子类，表示工具执行的结果：

```python
class ToolMessage(BaseMessage, ToolOutputMixin):
    tool_call_id: str       # 关联到哪个 ToolCall
    status: Literal["success", "error"] = "success"
    artifact: Any = None    # 不发送给模型的完整输出
```

`tool_call_id` 是关键——它将响应关联回请求。当模型并行调用多个工具时，这个 ID 让框架能正确配对每个工具的结果。

完整的调用循环：

```
用户消息 → LLM → AIMessage(tool_calls=[...])
                        │
                        ▼ 提取 ToolCall
                   工具执行 (BaseTool.invoke)
                        │
                        ▼
                   ToolMessage(tool_call_id=..., content=结果)
                        │
                        ▼
              [原始消息, AIMessage, ToolMessage] → LLM → 最终回答
```

`_format_output` 函数（`base.py` 第 1251 行）控制输出格式化：有 `tool_call_id` 时自动包装为 `ToolMessage`，否则直接返回原始内容。这就是为什么通过 `invoke(ToolCall)` 调用工具会得到 `ToolMessage`，而通过 `invoke("query string")` 调用会得到字符串。

## 工具在 LCEL 管道中的使用

因为工具是 Runnable，它可以直接参与 LCEL 管道组合：

```python
from langchain_core.output_parsers import StrOutputParser

# 简单管道：工具作为 Runnable 直接调用
chain = search_tool | StrOutputParser()
result = chain.invoke({"query": "LangChain"})
```

但更常见的模式是在模型输出后提取 ToolCall 并执行：

```python
from langchain_core.runnables import RunnableLambda

def call_tool(msg):
    """从 AIMessage 中提取 tool_calls 并执行。"""
    tool_map = {"web_search": search_tool}
    results = []
    for tc in msg.tool_calls:
        tool = tool_map[tc["name"]]
        results.append(tool.invoke(tc))  # 传入 ToolCall，得到 ToolMessage
    return results

chain = llm_with_tools | RunnableLambda(call_tool)
```

## 工具在 Agent 循环中的使用

Agent 循环与 LCEL 管道的区别在于：Agent 是**迭代**的。模型可能多次调用工具，每次调用的结果作为上下文反馈给模型，直到模型决定不再调用工具为止。

在 LangGraph Agent（v1 `create_agent`）中，工具执行发生在专门的图节点里。简化的循环逻辑：

```python
# 伪代码：Agent 循环中的工具调用
while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break  # 模型没有调用工具，结束循环

    for tool_call in response.tool_calls:
        tool = tool_map[tool_call["name"]]
        tool_msg = tool.invoke(tool_call)  # ToolCall → ToolMessage
        messages.append(tool_msg)
    # 带着 ToolMessage 继续循环
```

关键区别：LCEL 管道中工具是单次执行的变换节点；Agent 循环中工具是迭代反馈回路的一部分，模型根据工具返回的 `ToolMessage` 决定下一步行动。

## RetrieverTool：检索器到工具的桥接

`create_retriever_tool`（`langchain_core/tools/retriever.py`）是一个工厂函数，将 `BaseRetriever` 包装为 `StructuredTool`：

```python
from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=my_vector_store.as_retriever(),
    name="search_docs",
    description="搜索内部文档库",
    response_format="content_and_artifact",  # 同时返回文本和原始 Document
)
```

内部实现非常直接——定义一个接受 `query: str` 的函数，调用 `retriever.invoke(query)` 获取文档列表，用 `document_prompt` 格式化后拼接为字符串。输入 schema 固定为 `RetrieverInput`（只有一个 `query` 字段）。

这个桥接让检索器可以作为 Agent 的工具之一。模型不需要知道底层是向量搜索还是关键词检索——它只看到一个名为 `search_docs` 的工具，接受查询字符串，返回文本结果。

## InjectedToolArg 与运行时注入

`InjectedToolArg` 机制解决一个实际问题：有些参数是运行时需要的（如用户 ID、session 信息），但不应由 LLM 生成。

```python
from langchain_core.tools import tool, InjectedToolArg, InjectedToolCallId
from typing import Annotated

@tool
def get_user_orders(
    status: str,
    user_id: Annotated[str, InjectedToolArg],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """查询用户订单。"""
    return f"Orders for {user_id} with status={status} (call={tool_call_id})"
```

当生成 `tool_call_schema`（发送给 LLM 的 schema）时，`user_id` 和 `tool_call_id` 会被过滤掉。LLM 只看到 `status` 参数。但在工具实际执行时，框架通过 `_parse_input` 注入这些值——`tool_call_id` 从 `ToolCall.id` 提取，`user_id` 等自定义注入参数由调用方在 `tool_input` 字典中提供。

## 设计思考

### 为什么工具也是 Runnable

把工具设计为 Runnable 而非简单的 Callable，带来三个能力：

1. **统一组合**：工具可以用 `|` 操作符与 prompt、model、parser 组合成管道
2. **自动批处理和异步**：继承 `batch()`、`abatch()`、`astream()` 等方法
3. **可观测性**：自动接入 Callback/Tracer 系统，每次工具调用都有追踪

代价是引入了额外的抽象层。调试工具执行时，调用栈会穿过 `invoke` → `_prep_run_args` → `run` → `_to_args_and_kwargs` → `_parse_input` → `_run`，至少 5 层间接调用。

### ToolCall 协议如何统一不同供应商的函数调用格式

这是 LangChain 工具设计中最精妙的部分。不同 LLM 供应商的函数调用格式差异显著：

```
OpenAI:    {"function": {"name": "foo", "arguments": "{\"a\": 1}"}, "id": "call_xxx"}
Anthropic: {"type": "tool_use", "name": "foo", "input": {"a": 1}, "id": "toolu_xxx"}
Google:    {"functionCall": {"name": "foo", "args": {"a": 1}}}
```

参数可能是 JSON 字符串（OpenAI）或已解析的字典（Anthropic/Google）。ID 的格式和命名不同。有些支持并行调用，有些不支持。

LangChain 通过 `ToolCall` TypedDict 定义了一个最大公约数格式：`name` + `args`（已解析的字典）+ `id`。每个集成包（如 `langchain-openai`、`langchain-anthropic`）负责将供应商格式转换为标准 `ToolCall`。`default_tool_parser` 函数（`tool.py` 第 349 行）展示了 OpenAI 格式的转换：它解析 `function.arguments` JSON 字符串，处理解析失败的情况（生成 `InvalidToolCall`），并标准化 ID 字段。

这种设计的代价是：供应商特有的能力（如 Anthropic 的 `cache_control`、工具级别的 `defer_loading`）需要通过 `extras` 字段旁路传递，无法在标准协议中表达。`BaseTool` 的 `extras` 属性就是为此设计的逃生舱口。

### _format_output 的条件返回

`_format_output` 的行为值得注意：只有在存在 `tool_call_id` 时才包装为 `ToolMessage`。这意味着同一个工具在 LCEL 管道中直接调用（`invoke("query")`）和在 Agent 循环中通过 `ToolCall` 调用（`invoke({"name": "...", "args": {...}, "id": "..."})`）会返回不同类型的结果。这个设计让工具在两种使用模式中都能自然工作，但也要求开发者清楚自己处于哪种上下文中。

> **下一章**: [07 - 检索增强生成：从数据加载到语义检索的全链路](07-retrieval-and-rag.md)