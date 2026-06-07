# 04 - 模型与消息：LLM 交互的类型系统

## 为什么需要类型化的模型抽象

直接调用 OpenAI SDK，你拿到的是一个 JSON 字典。调用 Anthropic SDK，JSON 结构完全不同。当你的应用需要在多个供应商之间切换，或者在同一个管道中组合不同模型时，这些差异会变成大量的适配代码。

LangChain 的解法是建立一套**类型化的中间表示**：用统一的消息类型描述输入，用统一的 AIMessage 描述输出，用 Content Block 体系处理多模态数据。所有供应商适配器负责在这套中间表示和各自的 API 格式之间转换。

## 模型继承层级

```
RunnableSerializable[LanguageModelInput, LanguageModelOutputVar]
    │
    ├── BaseLanguageModel (ABC)        # 语言模型的通用基类
    │       ├── cache / verbose / callbacks / tags / metadata
    │       ├── generate_prompt()       # 抽象方法
    │       ├── with_structured_output() # 默认抛出 NotImplementedError
    │       ├── get_num_tokens()        # Token 计数（GPT-2 fallback）
    │       └── _get_ls_params()        # LangSmith 追踪参数
    │
    └── BaseChatModel(BaseLanguageModel[AIMessage])  # Chat 接口
            ├── rate_limiter            # 速率限制器
            ├── disable_streaming       # 流式控制
            ├── output_version          # "v0" | "v1" 内容格式
            ├── profile: ModelProfile   # 模型能力元数据
            ├── invoke() -> AIMessage   # 核心调用
            ├── stream() -> Iterator[AIMessageChunk]
            ├── bind_tools()            # 绑定工具
            ├── with_structured_output() # 结构化输出
            └── _generate()             # 子类必须实现的抽象方法
```

关键的类型签名在 `libs/core/langchain_core/language_models/base.py` 中：

```python
# 输入：可以是字符串、PromptValue、或消息列表
LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]

# 输出：消息或字符串
LanguageModelOutput = BaseMessage | str
```

`BaseLanguageModel` 继承自 `RunnableSerializable`，这意味着每个模型自动获得 Runnable 协议的全部能力（`invoke` / `stream` / `batch` / `ainvoke` 等）。`BaseChatModel` 将输出类型参数固定为 `AIMessage`，这是一个重要的类型约束。

## 消息类型系统

所有消息类型继承自 `BaseMessage`（`libs/core/langchain_core/messages/base.py`）：

```
BaseMessage (Serializable)
    ├── content: str | list[str | dict]   # 内容（支持多模态列表）
    ├── additional_kwargs: dict            # 供应商特定数据（如原始 tool_calls）
    ├── response_metadata: dict            # 响应头、logprobs、token 计数
    ├── type: str                          # 类型标识符，用于序列化
    ├── name: str | None                   # 可选名称
    ├── id: str | None                     # 唯一标识符
    ├── content_blocks -> list[ContentBlock]  # v1.0 标准化内容块（属性）
    └── text -> TextAccessor               # 便捷文本访问器
```

### 四种核心消息类型

```python
from langchain_core.messages import (
    SystemMessage,    # type="system"  —— 系统指令，设置模型行为
    HumanMessage,     # type="human"   —— 用户输入
    AIMessage,        # type="ai"      —— 模型响应
    ToolMessage,      # type="tool"    —— 工具执行结果
)
```

**SystemMessage** 和 **HumanMessage** 结构简单，主要是 `content` 字段。真正复杂的是 AIMessage 和 ToolMessage。

### AIMessage 的特殊性

AIMessage（`libs/core/langchain_core/messages/ai.py`）是最复杂的消息类型，因为它需要承载模型输出的所有信息：

```python
class AIMessage(BaseMessage):
    tool_calls: list[ToolCall] = []           # 标准化的工具调用
    invalid_tool_calls: list[InvalidToolCall] = []  # 解析失败的工具调用
    usage_metadata: UsageMetadata | None = None     # Token 用量统计
    type: Literal["ai"] = "ai"
```

`UsageMetadata` 提供了跨供应商统一的 token 计数：

```python
class UsageMetadata(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: NotRequired[InputTokenDetails]   # cache_read, cache_creation
    output_token_details: NotRequired[OutputTokenDetails]  # reasoning tokens
```

### ToolMessage 的协议角色

ToolMessage 是工具调用闭环的关键一环：

```python
class ToolMessage(BaseMessage):
    tool_call_id: str                              # 关联到哪个 tool_call
    status: Literal["success", "error"] = "success"
    artifact: Any = None                           # 不传递给模型的完整输出
```

`tool_call_id` 是必填字段，它将工具执行结果与 AIMessage 中的特定 `ToolCall` 关联起来。`artifact` 字段是一个聪明的设计：工具的完整输出可能很大（如完整的 API 响应），但只有摘要需要发送给模型，完整数据可以通过 `artifact` 保留给应用层使用。

### 流式消息：Chunk 变体

每种消息类型都有对应的 Chunk 变体（`AIMessageChunk`、`HumanMessageChunk` 等），支持通过 `+` 运算符累加：

```python
chunk1 = AIMessageChunk(content="Hello")
chunk2 = AIMessageChunk(content=" World")
full = chunk1 + chunk2  # AIMessageChunk(content="Hello World")
```

`AIMessageChunk` 额外持有 `tool_call_chunks`——工具调用的部分 JSON 片段，在流式结束时（`chunk_position="last"`）自动解析为完整的 `tool_calls`。

## Content Block 多模态体系

v1.0 引入的最重要的变化是 Content Block 体系（`libs/core/langchain_core/messages/content.py`）。它将 `content` 字段从简单的字符串扩展为一个结构化的内容块列表：

```python
# 所有内容块类型的联合
ContentBlock = (
    TextContentBlock          # {"type": "text", "text": "..."}
    | ReasoningContentBlock   # {"type": "reasoning", "reasoning": "..."}
    | ImageContentBlock       # {"type": "image", "url": "..." | "base64": "..."}
    | AudioContentBlock       # {"type": "audio", ...}
    | VideoContentBlock       # {"type": "video", ...}
    | FileContentBlock        # {"type": "file", ...}
    | PlainTextContentBlock   # {"type": "text-plain", ...}
    | ToolCall                # {"type": "tool_call", "name": "...", "args": {...}}
    | ToolCallChunk           # 流式工具调用片段
    | ServerToolCall          # 服务端执行的工具调用（如 web search）
    | ServerToolResult        # 服务端工具执行结果
    | InvalidToolCall         # 解析失败的工具调用
    | NonStandardContentBlock # 兜底：供应商特有的未标准化数据
)
```

每个内容块都是一个 TypedDict，共享统一的设计模式：

```python
class TextContentBlock(TypedDict):
    type: Literal["text"]              # 类型标识符
    id: NotRequired[str]               # 唯一标识
    text: str                          # 实际内容
    annotations: NotRequired[list[Annotation]]  # 引用标注
    index: NotRequired[int | str]      # 流式时的序号
    extras: NotRequired[dict[str, Any]]  # 供应商特有元数据
```

### 双版本输出：v0 与 v1

`BaseChatModel` 的 `output_version` 字段控制输出格式：

- **v0**（默认）：`content` 保持供应商原始格式，通过 `.content_blocks` 属性懒解析为标准块
- **v1**：`content` 直接存储标准化的内容块列表

这种双版本设计是为了**向后兼容**。现有代码可以继续用 `message.content` 拿到原始字符串，新代码可以用 `message.content_blocks` 获得结构化数据：

```python
msg = AIMessage(content="Paris is the capital of France.")

# v0 风格：直接访问字符串
print(msg.content)  # "Paris is the capital of France."

# v1 风格：结构化访问
print(msg.content_blocks)
# [{"type": "text", "text": "Paris is the capital of France."}]

# 便捷的 .text 属性（v1.0 新增）
print(msg.text)  # "Paris is the capital of France."
```

### 供应商格式的自动翻译

`content_blocks` 属性（`BaseMessage.content_blocks`）内部维护了一个多级翻译管道：

```python
# base.py 中的翻译链
for parsing_step in [
    _convert_v0_multimodal_input_to_v1,              # LangChain v0 格式
    _convert_to_v1_from_chat_completions_input,       # OpenAI 格式
    _convert_to_v1_from_anthropic_input,              # Anthropic 格式
    _convert_to_v1_from_genai_input,                  # Google GenAI 格式
    _convert_to_v1_from_converse_input,               # AWS Bedrock 格式
]:
    blocks = parsing_step(blocks)
```

无法识别的块会被包装为 `NonStandardContentBlock`，确保数据不丢失。

## init_chat_model：供应商无关的工厂

`init_chat_model`（`libs/langchain_v1/langchain/chat_models/base.py`）是 v1 提供的供应商无关初始化入口：

```python
from langchain.chat_models import init_chat_model

# 前缀格式：provider:model_name
model = init_chat_model("openai:gpt-4o", temperature=0)
model = init_chat_model("anthropic:claude-sonnet-4-5", max_tokens=1024)

# 也支持自动推断供应商
model = init_chat_model("gpt-4o")         # -> openai
model = init_chat_model("claude-sonnet-4-5")  # -> anthropic
```

内部实现是一个注册表 `_BUILTIN_PROVIDERS`，将供应商名映射到 `(module_path, class_name, creator_func)` 三元组，通过 `importlib` 延迟导入对应的集成包。截至当前版本，注册表包含 25+ 个供应商。

### 可配置模型

当不指定 `model` 参数时，`init_chat_model` 返回一个 `_ConfigurableModel`，可以在运行时通过 `config` 切换模型：

```python
configurable = init_chat_model(temperature=0)

# 运行时选择模型
configurable.invoke(
    "Hello",
    config={"configurable": {"model": "openai:gpt-4o"}}
)

# 切换到另一个供应商
configurable.invoke(
    "Hello",
    config={"configurable": {"model": "anthropic:claude-sonnet-4-5"}}
)
```

`_ConfigurableModel` 将声明式操作（如 `bind_tools`、`with_structured_output`）存入队列，在实际实例化模型时按顺序回放。

## ModelProfile：模型能力元数据

`ModelProfile`（`libs/core/langchain_core/language_models/model_profile.py`）是一个 TypedDict，描述模型的能力矩阵：

```python
class ModelProfile(TypedDict, total=False):
    name: str                    # 人类可读名称
    max_input_tokens: int        # 最大上下文窗口
    max_output_tokens: int       # 最大输出 token 数
    image_inputs: bool           # 是否支持图片输入
    audio_inputs: bool           # 是否支持音频输入
    video_inputs: bool           # 是否支持视频输入
    tool_calling: bool           # 是否支持工具调用
    structured_output: bool      # 是否支持原生结构化输出
    reasoning_output: bool       # 是否支持推理/思维链输出
    temperature: bool            # 是否支持 temperature 参数
    # ... 更多能力标志
```

Profile 数据来源于 [models.dev](https://github.com/sst/models.dev)，通过 `langchain-profiles` CLI 工具生成并存储在各集成包的 `data/` 目录中。`BaseChatModel` 在初始化时通过 `_resolve_model_profile()` 自动加载：

```python
@model_validator(mode="after")
def _set_model_profile(self) -> Self:
    if self.profile is None:
        with contextlib.suppress(Exception):
            self.profile = self._resolve_model_profile()
    return self
```

应用代码可以利用 profile 做能力检测：

```python
model = init_chat_model("openai:gpt-4o")
if model.profile and model.profile.get("image_inputs"):
    # 可以发送图片
    pass
```

## LangSmithParams：追踪参数

`LangSmithParams`（`libs/core/langchain_core/language_models/base.py`）是一个 TypedDict，专门用于向 LangSmith 追踪平台传递结构化的模型元数据：

```python
class LangSmithParams(TypedDict, total=False):
    ls_provider: str              # 供应商名（如 "openai"）
    ls_model_name: str            # 模型名（如 "gpt-4o"）
    ls_model_type: Literal["chat", "llm"]
    ls_temperature: float | None
    ls_max_tokens: int | None
    ls_stop: list[str] | None
    ls_integration: str           # 集成名（如 "langchain_chat_model"）
```

`BaseChatModel._get_ls_params()` 方法尝试从实例属性和调用参数中自动提取这些值。各集成包应该覆写此方法以提供准确的供应商名和模型名。

## with_structured_output：结构化输出

`with_structured_output()` 将模型包装为一个返回结构化数据的 Runnable：

```python
from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    temperature: float
    unit: str

model = init_chat_model("openai:gpt-4o")
structured = model.with_structured_output(Weather)
result = structured.invoke("What's the weather in Paris?")
# -> Weather(city="Paris", temperature=18.5, unit="celsius")
```

内部实现（`BaseChatModel.with_structured_output`）的核心逻辑：

1. 调用 `bind_tools([schema], tool_choice="any")` 强制模型使用工具调用
2. 根据 schema 类型选择解析器：Pydantic 类用 `PydanticToolsParser`，字典用 `JsonOutputKeyToolsParser`
3. 用 LCEL 管道组合：`llm | output_parser`

如果 `include_raw=True`，还会用 `RunnablePassthrough.assign` 和 `with_fallbacks` 构建一个带错误容忍的解析管道。

## 设计思考

### 为什么 Chat 接口成为事实标准

`BaseLanguageModel` 最初设计时，LLM API 有两种风格：completion（输入文本，输出文本）和 chat（输入消息列表，输出消息）。到 2024 年，几乎所有主流供应商都迁移到了 chat 接口——包括 OpenAI 废弃 completions API。`BaseChatModel` 将输出类型固定为 `AIMessage`（而非 `str`），这使得工具调用、usage 统计、response metadata 等结构化数据有了天然的承载位置。纯文本 completion 模型要返回 `tool_calls`，你得发明一种从文本中解析工具调用的协议；chat 接口原生支持。

### Content Block 如何应对多模态演进

Content Block 体系的核心设计决策是**使用 TypedDict 而非 Pydantic Model**。TypedDict 允许额外的键存在（通过 `extras` 字段），新的内容类型可以通过添加新的 TypedDict 来扩展联合类型，而无需修改现有代码。`NonStandardContentBlock` 作为兜底确保未知格式不会导致解析失败。这种设计让框架可以逐步跟进供应商的新能力（如 Google 的 thought signature、Anthropic 的 citations），而不需要同步发版。

另一个关键决策是 `content_blocks` 作为**属性而非字段**存在。这意味着标准化解析是惰性的，只在首次访问时执行。旧代码继续直接读取 `content` 字段不会受到任何影响。这是框架在保持向后兼容与推进标准化之间的平衡点。

### 工厂模式的取舍

`init_chat_model` 使用字符串注册表而非 importlib entry_points 或插件系统，优点是零魔法、容易调试；缺点是新供应商必须修改核心代码才能被注册。`_ConfigurableModel` 的声明式操作队列是一个巧妙但脆弱的设计——`bind_tools` 和 `with_structured_output` 被延迟执行，意味着参数验证也被延迟了。运行时切换供应商时，如果目标供应商不支持某个已排队的操作，错误会在调用时才暴露。

> **下一章**: [05 - Prompt 模板与输出解析：模型 I/O 管线](05-prompts-and-output-parsing.md)