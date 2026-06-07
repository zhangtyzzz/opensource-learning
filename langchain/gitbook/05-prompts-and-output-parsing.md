# 05 - Prompt 模板与输出解析：模型 I/O 管线

LLM 调用可以被拆解为三个阶段：**格式化输入**、**调用模型**、**解析输出**。上一章讨论了中间环节——模型与消息的类型系统。本章聚焦两端：Prompt 模板如何将原始变量组装成模型能理解的输入，Output Parser 如何将模型的自由文本转化为程序能处理的结构化数据。

```
dict (变量)                              str | BaseMessage
    │                                         │
    ▼                                         ▼
┌──────────────────┐    ┌─────────┐    ┌──────────────────┐
│ BasePromptTemplate│───>│  Model  │───>│ BaseOutputParser │
│ Runnable[dict,    │    │         │    │ Runnable[str|Msg,│
│    PromptValue]   │    │         │    │         T]       │
└──────────────────┘    └─────────┘    └──────────────────┘
    PromptValue                              T (结构化)
```

这条管线在 LCEL 中可以用 `|` 一行表达：

```python
chain = prompt | model | parser
```

三个组件都是 Runnable，类型签名首尾相接，构成一条类型安全的管道。

---

## Prompt 模板体系

### BasePromptTemplate：模板即 Runnable

所有 Prompt 模板的基类定义在 `libs/core/langchain_core/prompts/base.py`：

```python
class BasePromptTemplate(
    RunnableSerializable[dict, PromptValue], ABC, Generic[FormatOutputType]
):
    input_variables: list[str]
    optional_variables: list[str] = []
    partial_variables: Mapping[str, Any] = {}
    output_parser: BaseOutputParser | None = None
```

类型签名 `Runnable[dict, PromptValue]` 清晰地表达了模板的契约：接收一个变量字典，产出一个 `PromptValue`。`PromptValue` 是模板和模型之间的桥梁——它同时实现了 `to_string()` 和 `to_messages()`，让同一个模板既能对接纯文本 LLM，也能对接 Chat 模型。

`invoke()` 的实现揭示了模板作为 Runnable 的工作方式：

```python
def invoke(self, input: dict, config=None, **kwargs) -> PromptValue:
    config = ensure_config(config)
    return self._call_with_config(
        self._format_prompt_with_error_handling,
        input,
        config,
        run_type="prompt",          # 追踪系统中的类型标记
        serialized=self._serialized,
    )
```

`_call_with_config` 来自 Runnable 基类（第三章详述），它自动处理回调、追踪和错误上报。这意味着每次模板格式化都会被 LangSmith 等追踪工具记录为一次 `prompt` 类型的 run。

### 变量插值：三种语法

LangChain 支持三种模板语法，定义为字面量类型：

```python
PromptTemplateFormat = Literal["f-string", "mustache", "jinja2"]
```

**f-string（默认）**——使用 Python 原生格式化语法：

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Tell me about {topic} in {language}")
prompt.format(topic="Python", language="Chinese")
# -> "Tell me about Python in Chinese"
```

变量提取通过 `string.Formatter().parse()` 实现。f-string 语法有明确的安全限制——禁止属性访问（`{obj.attr}`）和索引操作（`{arr[0]}`），这是刻意的安全设计。

**Mustache**——使用双花括号语法，通过内置的 `langchain_core.utils.mustache` 模块解析：

::: v-pre
```python
prompt = PromptTemplate.from_template(
    "Hello {{name}}, you are {{age}} years old",
    template_format="mustache",
)
```
:::

Mustache 支持嵌套对象和节语法，适合结构化数据较多的场景。它的变量提取通过 tokenizer 实现，只取顶层 key。

**Jinja2**——最强大但也最危险：

::: v-pre
```python
prompt = PromptTemplate.from_template(
    "{% for item in items %}{{ item }}{% endfor %}",
    template_format="jinja2",
)
```
:::

Jinja2 使用 `SandboxedEnvironment` 渲染以防止沙箱逃逸，但源码中反复强调：**绝不要接受不受信任来源的 Jinja2 模板**。这是安全与灵活性的典型权衡。

### ChatPromptTemplate：消息模板组合

`ChatPromptTemplate` 是现代 LangChain 中最常用的模板类，定义在 `libs/core/langchain_core/prompts/chat.py`。它将多条消息模板组合成一个完整的对话结构：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}."),
    ("human", "{question}"),
])

prompt.invoke({"name": "Alice", "question": "What is LangChain?"})
# -> ChatPromptValue(messages=[
#     SystemMessage(content="You are a helpful assistant named Alice."),
#     HumanMessage(content="What is LangChain?"),
# ])
```

消息的表达方式非常灵活。`_convert_to_message_template()` 函数支持五种输入格式：

| 格式 | 示例 |
|---|---|
| `BaseMessagePromptTemplate` 实例 | `HumanMessagePromptTemplate(...)` |
| `BaseMessage` 实例 | `SystemMessage(content="hello")` |
| `(type, template)` 二元组 | `("human", "{input}")` |
| `(class, template)` 二元组 | `(HumanMessage, "{input}")` |
| 纯字符串（默认 human） | `"{input}"` |

消息类型字符串到模板类的映射关系：

```python
"human" / "user"      -> HumanMessagePromptTemplate
"ai" / "assistant"    -> AIMessagePromptTemplate
"system"              -> SystemMessagePromptTemplate
"placeholder"         -> MessagesPlaceholder
```

### MessagesPlaceholder：动态消息注入

`MessagesPlaceholder` 是处理聊天历史的关键机制。它不格式化单条消息，而是将一个完整的消息列表注入到模板中：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{history}"),  # 语法糖，等价于下面的写法
    # MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}"),
])
```

几个关键设计点：

1. **`optional=True` 的自动推断**：当使用 `("placeholder", "{var}")` 语法时，`optional` 自动设为 `True`，变量被加入 `partial_variables` 中（默认空列表）。这意味着即使不传 `history`，模板也能正常工作。

2. **`n_messages` 截断**：`MessagesPlaceholder` 支持 `n_messages` 参数，只保留最后 N 条消息。这是一种简单的上下文窗口管理策略。

3. **类型标注**：Placeholder 变量被自动标注为 `list[AnyMessage]` 类型，这使得 `get_input_schema()` 能生成正确的 Pydantic 模型。

### ChatPromptTemplate 的组合操作

`ChatPromptTemplate` 支持 `+` 运算符和 `__getitem__` 切片，实现模板的代数式组合：

```python
system = ChatPromptTemplate.from_messages([("system", "You are {role}.")])
user = ChatPromptTemplate.from_messages([("human", "{input}")])
combined = system + user  # 消息列表拼接

# 切片返回新的 ChatPromptTemplate
first_two = combined[:2]
```

`partial()` 方法允许预填充部分变量，返回一个新的模板实例：

```python
template = prompt.partial(name="Alice")
# 后续调用只需要传 question
template.invoke({"question": "Hi!"})
```

---

## Few-Shot Prompting

### FewShotPromptTemplate

Few-shot prompting 是提升 LLM 输出质量的核心技术。LangChain 通过 `_FewShotPromptTemplateMixin` 统一了静态示例和动态选择两种模式：

```python
class _FewShotPromptTemplateMixin(BaseModel):
    examples: list[dict] | None = None           # 静态示例列表
    example_selector: BaseExampleSelector | None = None  # 动态选择器
```

校验器确保二者必须且只能提供一个。

`FewShotPromptTemplate`（用于纯文本）的 `format()` 逻辑清晰地展示了模板组装过程：

```python
def format(self, **kwargs):
    examples = self._get_examples(**kwargs)
    example_strings = [self.example_prompt.format(**ex) for ex in examples]
    pieces = [self.prefix, *example_strings, self.suffix]
    template = self.example_separator.join(pieces)
    return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)
```

对于 Chat 模型场景，`FewShotChatMessagePromptTemplate` 将每个示例格式化为一组消息，然后嵌入到 `ChatPromptTemplate` 中：

```python
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {input}?"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor."),
    few_shot,
    ("human", "What is {input}?"),
])
```

### ExampleSelector

`BaseExampleSelector` 定义了动态示例选择的接口：

```python
class BaseExampleSelector(ABC):
    def add_example(self, example: dict[str, str]) -> Any: ...
    def select_examples(self, input_variables: dict[str, str]) -> list[dict]: ...
```

内置实现包括：

- **`LengthBasedExampleSelector`**——根据 token 长度限制选择示例，确保不超出上下文窗口
- **`SemanticSimilarityExampleSelector`**——通过向量相似度选择与当前输入最相关的示例
- **`MaxMarginalRelevanceExampleSelector`**——在相关性和多样性之间取平衡（MMR 算法）

---

## 输出解析体系

### BaseOutputParser：从文本到结构

Output Parser 的基类定义在 `libs/core/langchain_core/output_parsers/base.py`：

```python
class BaseOutputParser(
    BaseLLMOutputParser, RunnableSerializable[LanguageModelOutput, T]
):
    def parse(self, text: str) -> T: ...              # 核心：文本 -> 结构化
    def parse_result(self, result: list[Generation]) -> T: ...
    def get_format_instructions(self) -> str: ...     # 告诉 LLM 如何格式化输出
```

类型签名 `Runnable[LanguageModelOutput, T]` 表明解析器接受模型输出（`str | BaseMessage`），产出泛型 `T`。`invoke()` 方法根据输入类型自动选择处理路径：

```python
def invoke(self, input: str | BaseMessage, config=None, **kwargs) -> T:
    if isinstance(input, BaseMessage):
        return self._call_with_config(
            lambda inner: self.parse_result([ChatGeneration(message=inner)]),
            input, config, run_type="parser",
        )
    return self._call_with_config(
        lambda inner: self.parse_result([Generation(text=inner)]),
        input, config, run_type="parser",
    )
```

注意 `run_type="parser"` 标记——这使得追踪系统能区分解析步骤和其他步骤。

### 流式解析的分层设计

输出解析器有两个流式基类（`libs/core/langchain_core/output_parsers/transform.py`）：

```
BaseOutputParser[T]
    └── BaseTransformOutputParser[T]       # 逐 chunk 独立解析
        └── BaseCumulativeTransformOutputParser[T]  # 累积 chunk 后解析
```

`BaseTransformOutputParser` 对每个 chunk 独立调用 `parse_result()`——适合 `StrOutputParser` 这种透传场景。

`BaseCumulativeTransformOutputParser` 更有趣：它将所有 chunk 累积成一个 `GenerationChunk`，然后用 `partial=True` 模式解析。只有当解析结果变化时才 yield 新值。这使得 `JsonOutputParser` 能在流式模式下渐进式地返回部分 JSON 对象。

### 内置解析器速查

#### StrOutputParser

最简单的解析器，直接返回输入文本：

```python
class StrOutputParser(BaseTransformOutputParser[str]):
    def parse(self, text: str) -> str:
        return text
```

几乎所有 LCEL 链的终端都会用到它，尤其是需要流式输出的场景。

#### JsonOutputParser

解析 LLM 输出中的 JSON（支持 Markdown 代码块包裹）：

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
parser.invoke('```json\n{"name": "Alice", "age": 30}\n```')
# -> {"name": "Alice", "age": 30}
```

可以通过 `pydantic_object` 参数提供 schema 约束，`get_format_instructions()` 会将 schema 注入 Prompt 中引导 LLM 输出。流式模式下使用 `parse_partial_json` 解析不完整的 JSON 片段。

#### PydanticOutputParser

在 `JsonOutputParser` 基础上增加 Pydantic 校验：

```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)
parser.invoke('{"name": "Alice", "age": 30}')
# -> Person(name='Alice', age=30)
```

继承链为 `PydanticOutputParser -> JsonOutputParser -> BaseCumulativeTransformOutputParser`，先解析 JSON，再通过 `model_validate()` 转为 Pydantic 对象。

#### XMLOutputParser

适合 Anthropic Claude 等对 XML 格式更擅长的模型：

```python
from langchain_core.output_parsers import XMLOutputParser

parser = XMLOutputParser(tags=["person", "name", "age"])
parser.invoke("<person><name>Alice</name><age>30</age></person>")
# -> {"person": [{"name": "Alice"}, {"age": "30"}]}
```

流式解析使用 `ET.XMLPullParser` 实现增量 XML 解析。安全方面默认使用 `defusedxml` 库防止 XML 实体注入攻击。

#### PydanticToolsParser（OpenAI 专用）

直接从 OpenAI 的 tool_calls 响应中提取结构化数据，绕过文本解析：

```python
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

parser = PydanticToolsParser(tools=[Person])
# 直接从 AIMessage.tool_calls 中提取并校验
```

这是最可靠的结构化输出方式——它利用模型原生的 function calling / tool use 能力，不依赖文本解析。内部通过检查 `AIMessage.tool_calls` 字段获取已解析的调用参数，然后用 Pydantic 模型校验。

### 选择解析器的决策树

```
需要结构化输出？
├── 否 → StrOutputParser
└── 是
    ├── 模型支持 tool_calls？
    │   └── 是 → PydanticToolsParser（最可靠）
    └── 否
        ├── 需要 Pydantic 校验？
        │   └── 是 → PydanticOutputParser
        ├── 只需要 JSON？
        │   └── 是 → JsonOutputParser
        └── 模型擅长 XML？
            └── 是 → XMLOutputParser
```

值得注意的是，`output_parsers/__init__.py` 的模块文档中明确指出：如今大多数 LLM 原生支持结构化输出（如 `with_structured_output()`），在这种情况下 Output Parser 可能并非必需。Output Parser 的价值主要体现在：模型不支持原生结构化输出时，或需要对输出做额外处理和校验时。

---

## 设计思考

### 为什么 Prompt 模板本身也是 Runnable？

将 `BasePromptTemplate` 设计为 `Runnable[dict, PromptValue]` 看似只是为了能用 `|` 运算符，但实际意义更深远：

**统一的执行语义**。模板格式化自动获得 `invoke`/`ainvoke`/`batch`/`stream` 四套方法，以及完整的回调和追踪支持。在 LangSmith 中，模板格式化被记录为一次独立的 `run_type="prompt"` 的执行，你能看到传入了什么变量、产出了什么 PromptValue。

**可组合性**。模板可以像任何 Runnable 一样参与管道组合。`prompt | model | parser` 不是语法糖——它创建了一个 `RunnableSequence`，其中每个节点的类型约束由泛型参数保证：`dict -> PromptValue -> BaseMessage -> T`。

**可替换性**。因为模板是 Runnable，你可以用 `RunnableLambda(lambda x: ...)` 替换模板，用自定义函数动态生成 Prompt，而不破坏管道的其余部分。

### PromptValue 的桥梁角色

`PromptValue` 是一个精巧的抽象。它同时实现 `to_string()` 和 `to_messages()`，使得同一个模板既能对接 `BaseLLM`（纯文本补全）也能对接 `BaseChatModel`（消息对话）。这个设计来自 LangChain 早期需要同时支持两种模型接口的历史需求。随着 Chat 模型成为事实标准，`PromptValue` 的桥梁角色变得不那么关键，但它作为模板和模型之间的类型中间层依然有价值。

### 输出解析器的历史定位

Output Parser 体系在源码注释中被标记为"早期解决方案"。随着模型原生 structured output 能力的成熟（OpenAI 的 function calling、Anthropic 的 tool use），基于文本解析的方式正在让位于基于 schema 约束的方式（即 `BaseChatModel.with_structured_output()`）。

但 Output Parser 的设计仍然值得学习：它展示了如何用 Runnable 接口将一个"解析"步骤无缝嵌入管道，以及如何通过 `get_format_instructions()` 在输入端（Prompt）和输出端（Parser）之间建立协议——Parser 不仅解析输出，还能生成指令告诉 LLM 应该如何格式化输出。这种输入输出的双向协调是 LangChain I/O 管线的核心洞见。

> **下一章**: [06 - 工具与函数调用：扩展 LLM 的行动能力](06-tools-and-tool-calling.md)