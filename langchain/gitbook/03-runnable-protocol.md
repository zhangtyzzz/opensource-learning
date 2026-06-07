# 03 - Runnable 协议：LCEL 的统一计算模型

上一章我们看到 LangChain 的分层包架构。本章深入 `langchain-core` 中最核心的抽象——`Runnable` 协议。这个 6574 行的基类（`runnables/base.py`）定义了 LangChain 的统一计算模型：任何可被调用、批处理、流式输出和组合的计算单元，都是一个 Runnable。理解它，就理解了 LangChain Expression Language（LCEL）的全部底层机制。

## Runnable[Input, Output] 泛型接口

`Runnable` 是一个带有两个类型参数的泛型抽象基类：

```python
# langchain_core/runnables/base.py
class Runnable(ABC, Generic[Input, Output]):
    """A unit of work that can be invoked, batched, streamed, 
    transformed and composed."""
    
    name: str | None
    
    @abstractmethod
    def invoke(self, input: Input, config: RunnableConfig | None = None, **kwargs) -> Output:
        """Transform a single input into an output."""
    
    async def ainvoke(self, input: Input, config: RunnableConfig | None = None, **kwargs) -> Output:
        """Async version - defaults to running invoke in executor."""
        return await run_in_executor(config, self.invoke, input, config, **kwargs)
```

这个设计的关键洞察：`Input` 和 `Output` 是**完全开放**的类型参数。一个 Prompt Template 是 `Runnable[dict, PromptValue]`，一个 Chat Model 是 `Runnable[PromptValue, BaseMessage]`，一个 Output Parser 是 `Runnable[BaseMessage, str]`。类型参数让 IDE 可以在管道组合时做静态检查（虽然实际的 runtime 类型常常是 `Any`）。

`Runnable` 继承了 `ABC` 和 `Generic[Input, Output]`，同时通过 Pydantic 的 `BaseModel`（经由 `RunnableSerializable`）获得了序列化能力。这意味着整个 Runnable 树可以被 JSON 序列化和反序列化——一个为 LangSmith 追踪设计的特性。

## 六个统一方法：invoke / stream / batch

每个 Runnable 都自动拥有六个核心方法，分为同步和异步两组：

| 同步方法 | 异步方法 | 语义 |
|---------|---------|------|
| `invoke(input)` | `ainvoke(input)` | 单输入 -> 单输出 |
| `stream(input)` | `astream(input)` | 单输入 -> 输出流（Iterator/AsyncIterator） |
| `batch(inputs)` | `abatch(inputs)` | 多输入 -> 多输出（并行） |

**默认实现的降级策略**是理解这套 API 的关键：

```python
# ainvoke 默认：在线程池中跑 invoke
async def ainvoke(self, input, config=None, **kwargs):
    return await run_in_executor(config, self.invoke, input, config, **kwargs)

# batch 默认：用 ThreadPoolExecutor 并行跑 invoke
def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
    configs = get_config_list(config, len(inputs))
    with get_executor_for_config(configs[0]) as executor:
        return list(executor.map(
            lambda input_, config: self.invoke(input_, config, **kwargs),
            inputs, configs
        ))

# stream 默认：调 invoke，yield 整个结果（无真正流式）
def stream(self, input, config=None, **kwargs):
    yield self.invoke(input, config, **kwargs)
```

这意味着**你只需实现 `invoke`，就自动获得异步、流式和批处理能力**。但性能敏感的子类应该覆写：Chat Model 覆写 `stream` 来逐 token 输出，Embedding Model 覆写 `batch` 来利用 API 批量接口。

## RunnableSequence 与 pipe `|` 运算符

`RunnableSequence` 是 LCEL 最重要的组合原语——它将多个 Runnable 串联成一个管道，前一步的输出是后一步的输入：

```python
from langchain_core.runnables import RunnableLambda

# 使用 | 运算符构建序列
chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
chain.invoke(3)  # (3 + 1) * 2 = 8

# 等价于显式构造
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(RunnableLambda(lambda x: x + 1), RunnableLambda(lambda x: x * 2))
```

`|` 运算符通过 `__or__` 和 `__ror__` 实现。源码中 `__or__` 接受的参数类型非常宽泛：

```python
def __or__(self, other: 
    Runnable[Any, Other]                                    # 另一个 Runnable
    | Callable[[Iterator[Any]], Iterator[Other]]            # 流式转换函数
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]  # 异步流式转换
    | Callable[[Any], Other]                                # 普通函数
    | Mapping[str, ...]                                     # dict -> RunnableParallel
) -> RunnableSerializable[Input, Other]:
    return RunnableSequence(self, coerce_to_runnable(other))
```

`coerce_to_runnable` 函数负责类型转换：普通函数变成 `RunnableLambda`，dict 变成 `RunnableParallel`。这就是为什么你可以直接写 `prompt | model | parser` —— 三种完全不同类型的对象被统一为 Runnable 后通过管道串联。

**RunnableSequence 的内部结构**使用 `first`/`middle`/`last` 三段式存储，这是为了类型推导服务的：

```python
class RunnableSequence(RunnableSerializable[Input, Output]):
    first: Runnable[Input, Any]          # 输入类型来自 first
    middle: list[Runnable[Any, Any]]     # 中间步骤类型开放
    last: Runnable[Any, Output]          # 输出类型来自 last
```

`invoke` 实现是一个简单循环，关键在于每一步都通过 `run_manager.get_child()` 创建子回调上下文：

```python
def invoke(self, input, config=None, **kwargs):
    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
    run_manager = callback_manager.on_chain_start(None, input, name=self.get_name())
    
    try:
        for i, step in enumerate(self.steps):
            config = patch_config(config, callbacks=run_manager.get_child(f"seq:step:{i+1}"))
            input = step.invoke(input, config)
    except BaseException as e:
        run_manager.on_chain_error(e)
        raise
    else:
        run_manager.on_chain_end(input)
        return input
```

## RunnableParallel 与 dict 字面量组合

`RunnableParallel` 是第二个核心组合原语——它接收同一输入，并行执行多个 Runnable，输出一个字典：

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 显式构造
parallel = RunnableParallel(
    doubled=RunnableLambda(lambda x: x * 2),
    tripled=RunnableLambda(lambda x: x * 3),
)
parallel.invoke(5)  # {"doubled": 10, "tripled": 15}

# 更常见的用法：dict 字面量在管道中自动转为 RunnableParallel
chain = RunnableLambda(lambda x: x + 1) | {
    "doubled": RunnableLambda(lambda x: x * 2),
    "tripled": RunnableLambda(lambda x: x * 3),
}
chain.invoke(4)  # {"doubled": 10, "tripled": 15}
```

RunnableParallel 的签名是 `Runnable[Input, dict[str, Any]]`——输出始终是一个 dict。在实际的 RAG 链中，它常被用来构造 prompt 输入：

```python
from langchain_core.runnables import RunnablePassthrough

# 典型 RAG 模式：并行获取 context 和透传 question
chain = {
    "context": retriever,                          # Runnable[str, list[Document]]
    "question": RunnablePassthrough(),             # Runnable[str, str] 
} | prompt | model | parser
```

这里 dict `{"context": retriever, "question": RunnablePassthrough()}` 被 `coerce_to_runnable` 自动转换为 `RunnableParallel`。

## 辅助 Runnable 类型

### RunnableLambda

将任意 Python 函数包装为 Runnable。支持同时提供同步和异步实现：

```python
from langchain_core.runnables import RunnableLambda

def sync_fn(x: int) -> int:
    return x + 1

async def async_fn(x: int) -> int:
    return x + 1

runnable = RunnableLambda(sync_fn, afunc=async_fn)
```

一个重要特性：如果 `RunnableLambda` 返回另一个 `Runnable`，该 Runnable 会被自动调用。这允许动态路由——在运行时决定执行哪个子链。

### RunnableBranch

基于条件选择执行路径的路由器：

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: isinstance(x, str), lambda x: x.upper()),
    (lambda x: isinstance(x, int), lambda x: x + 1),
    lambda x: "default",  # 最后一个参数是默认分支
)

branch.invoke("hello")  # "HELLO"
branch.invoke(42)        # 43
branch.invoke(None)      # "default"
```

内部实现就是顺序执行条件函数，第一个返回 `True` 的条件对应的 Runnable 被选中。如果全部不满足，走默认分支。

### RunnablePassthrough

身份函数的 Runnable 包装，直接透传输入。但它有一个关键的 `.assign()` 方法：

```python
from langchain_core.runnables import RunnablePassthrough

# 基本透传
passthrough = RunnablePassthrough()
passthrough.invoke("hello")  # "hello"

# assign：保留原始输入（dict），并添加新的键值对
chain = RunnablePassthrough.assign(
    uppercased=lambda x: x["text"].upper()
)
chain.invoke({"text": "hello"})  # {"text": "hello", "uppercased": "HELLO"}
```

`RunnablePassthrough.assign()` 等价于 `RunnablePassthrough() | RunnableParallel(原始键 + 新键)`，它是构建数据流管道时最常用的"添加字段"操作。

## RunnableConfig 上下文传播

`RunnableConfig` 是一个 `TypedDict`，定义了贯穿整个执行链的运行时配置：

```python
class RunnableConfig(TypedDict, total=False):
    tags: list[str]                # 标签（用于过滤追踪）
    metadata: dict[str, Any]       # 元数据（传递给追踪系统）
    callbacks: Callbacks           # 回调处理器列表
    run_name: str                  # 追踪 run 名称
    max_concurrency: int | None    # 最大并行度
    recursion_limit: int           # 递归深度限制（默认 25）
    configurable: dict[str, Any]   # 运行时可配置项
    run_id: uuid.UUID | None       # 唯一执行 ID
```

`total=False` 意味着所有字段都是可选的。配置通过 `ContextVar` 自动在父子 Runnable 之间传播：

```python
# 父链设置 tags
chain.invoke(input, config={"tags": ["production"], "metadata": {"user_id": "123"}})

# 子链通过 ensure_config 自动继承父链的 tags 和 metadata
# ensure_config({"tags": ["child"]}) -> {"tags": ["production", "child"], ...}
```

配置传播是通过 `patch_config` 和 `merge_configs` 实现的。`merge_configs` 的合并策略是：`tags` 和 `metadata` 做**追加合并**，`callbacks` 做**列表合并**，其他字段做**后者覆盖**。这确保了追踪标签不会在传播过程中丢失。

## Fallbacks 与重试策略

### with_retry：指数退避重试

```python
from langchain_core.runnables import RunnableLambda

# 对不稳定的 API 调用添加重试
robust_chain = unstable_runnable.with_retry(
    retry_if_exception_type=(TimeoutError, ConnectionError),
    stop_after_attempt=3,
    wait_exponential_jitter=True,  # 使用 tenacity 的指数退避 + 抖动
)
```

`with_retry` 返回一个 `RunnableRetry` 实例（定义在 `runnables/retry.py`，379 行），内部使用 `tenacity` 库实现重试逻辑。重试发生在单个 Runnable 层面，不会回退到链的上游重新执行。

### with_fallbacks：备选方案链

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 主模型失败时自动切换到备选模型
model = ChatOpenAI(model="gpt-4o").with_fallbacks(
    [ChatAnthropic(model="claude-sonnet-4-20250514")],
    exceptions_to_handle=(Exception,),
)
```

`with_fallbacks` 返回 `RunnableWithFallbacks`（定义在 `runnables/fallbacks.py`，664 行）。它按顺序尝试主 Runnable 和每个 fallback，直到其中一个成功。`exception_key` 参数允许将捕获的异常信息传递给 fallback，让 fallback 可以根据错误类型做不同处理。

## 组合示例：一个完整的 RAG 链

将以上所有概念组合在一起，一个典型的 RAG 链长这样：

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Based on the context: {context}\nAnswer: {question}"
)
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# 整条链的类型流：str -> dict -> PromptValue -> BaseMessage -> str
chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt     # Runnable[dict, PromptValue]
    | model      # Runnable[PromptValue, BaseMessage]
    | parser     # Runnable[BaseMessage, str]
)

# 一条链，自动获得所有执行模式
result = chain.invoke("What is LangChain?")           # 同步
result = await chain.ainvoke("What is LangChain?")    # 异步
for chunk in chain.stream("What is LangChain?"):      # 流式
    print(chunk, end="")
```

这条链在内存中是一个嵌套的 Runnable 树：最外层是 `RunnableSequence`，第一步是 `RunnableParallel`（从 dict 转换），其中一个分支内部又嵌套了 `RunnableSequence`（`retriever | format_docs`）。

## 设计思考

### 为什么选择 Unix pipe 而不是 DAG？

LangChain 的 LCEL 选择了**线性管道**（`|` 运算符）作为核心组合方式，而不是 DAG（有向无环图）。这个选择有明确的工程考量：

**管道组合的优势：**
- 语法极其简洁：`prompt | model | parser` 一行表达完整链
- 心智模型简单：数据从左到右线性流动，和 Unix shell 管道完全一致
- 自动流式穿透：`RunnableSequence.stream` 调用 `transform` 链式传播，每一步产出的 chunk 立即传给下一步

**通过 RunnableParallel 扩展为有限 DAG：**
- dict 字面量让分叉变得自然：`{"a": chain_a, "b": chain_b}` 是一个扇出节点
- 但只支持扇出再汇合（汇合点就是 dict 的输出），不支持任意 DAG 拓扑
- 如果需要真正的 DAG，LangChain 提供了 LangGraph 作为专门的解决方案

### 代价：调试困难与调用栈深度

这套设计并非没有代价：

1. **调用栈深度**：一条 5 步的 LCEL 链，实际调用栈可能有 20+ 层。每个 Runnable 的 `invoke` 都经过 `ensure_config` -> `get_callback_manager_for_config` -> `on_chain_start` -> `set_config_context` -> 实际执行 -> `on_chain_end`。当链嵌套链时，栈深度呈指数增长。`recursion_limit=25` 是一个硬性保护。

2. **错误信息不直观**：当链中间某步类型不匹配时，你得到的不是"step 3 expected dict but got str"，而是 Pydantic 验证错误或者嵌套在多层 `RunnableSequence` 中的 `TypeError`。

3. **隐式类型转换**：`coerce_to_runnable` 会自动将函数和 dict 转换为 Runnable，这在写代码时很方便，但在调试时会让你困惑于"我的函数为什么被包了三层 RunnableLambda"。

4. **流式语义不统一**：有些 Runnable（如 `StrOutputParser`）可以流式处理输入 chunk，有些（如复杂的 `RunnableLambda`）只能等到完整输入后才能产出。在一条链中混用两种模式时，流式的 latency 特性变得不可预测。

这些代价直接推动了 Chapter 09 中可观测性系统的设计——没有 tracing，调试 LCEL 链几乎不可能。

### 为什么是 6574 行的基类？

`base.py` 之所以如此庞大，是因为 `Runnable` 基类承载了大量**默认行为**：

- 六个执行方法的默认实现（sync/async 互转、batch 并行化）
- `astream_events` / `astream_log` 等高级流式 API
- `with_retry` / `with_fallbacks` / `with_config` / `with_listeners` 等装饰方法
- `input_schema` / `output_schema` / `config_schema` 等 schema 推导
- `get_graph()` 用于可视化链结构

这是一个典型的**Template Method 模式**：基类定义了执行的骨架（包括回调管理、配置传播、错误处理），子类只需覆写核心计算逻辑。代价是这个基类成为了一个上帝类（God Class），任何修改都可能影响所有 Runnable 实现。

---

> 源码路径：
> - `libs/core/langchain_core/runnables/base.py` -- Runnable, RunnableSequence, RunnableParallel, RunnableLambda (6574 行)
> - `libs/core/langchain_core/runnables/config.py` -- RunnableConfig, ensure_config, merge_configs (672 行)
> - `libs/core/langchain_core/runnables/passthrough.py` -- RunnablePassthrough (841 行)
> - `libs/core/langchain_core/runnables/branch.py` -- RunnableBranch (461 行)
> - `libs/core/langchain_core/runnables/fallbacks.py` -- RunnableWithFallbacks (664 行)
> - `libs/core/langchain_core/runnables/retry.py` -- RunnableRetry (379 行)

[04 - 模型与消息](04-models-and-messages.md)
