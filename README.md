# Open Source Agent Framework Learning

Agent / LLM 相关开源框架深度学习仓库。不只是"会用"，而是理解设计决策、权衡取舍和实现细节。

## 目录结构

```
opensource-learning/
├── langgraph/              # LangGraph 框架
│   ├── source/             # 源码（git submodule）
│   ├── gitbook/            # 系统化框架解读文档（像一本书）
│   │   ├── diagrams/       # Excalidraw 架构图 + PNG
│   │   ├── 00-index.md     # 目录索引
│   │   ├── 01-overview.md  # 框架概览
│   │   ├── 02-architecture.md
│   │   └── ...
│   ├── notes/              # 问题研究笔记（生产实践中遇到的问题）
│   └── examples/           # 自己写的示例代码
├── langchain/
│   ├── source/
│   ├── gitbook/
│   ├── notes/
│   └── examples/
├── mem0/
│   ├── source/
│   ├── gitbook/
│   ├── notes/
│   └── examples/
└── ...
```

## 每个框架目录的约定

| 子目录 | 用途 | 内容特点 |
|--------|------|---------|
| `source/` | 框架源码（git submodule） | 只读，跟踪上游特定版本 |
| `gitbook/` | 系统化框架解读 | 借鉴官方文档 + 深度分析 + 架构图，像一本完整的参考书 |
| `notes/` | 问题研究笔记 | 官方文档未解答的问题、生产踩坑、源码走读发现 |
| `examples/` | 实验代码 | 验证理解、复现问题、测试方案 |

### gitbook vs notes 的区别

- **gitbook** = "教科书"：系统化、有章节结构、覆盖框架全貌、配架构图
- **notes** = "实验笔记"：问题驱动、深挖某个具体点、记录真实发现

## 源码管理方式

使用 **git submodule** 引入各框架源码：

```bash
# 添加 submodule（已执行的）
git submodule add https://github.com/langchain-ai/langgraph.git langgraph/source
git submodule add https://github.com/langchain-ai/langchain.git langchain/source
git submodule add https://github.com/mem0ai/mem0.git mem0/source

# clone 本仓库时拉取所有 submodule
git clone --recursive <repo-url>

# 已 clone 后补充拉取
git submodule update --init --recursive
```

## 学习进度

### LangGraph
- [x] Gitbook: 架构总览图
- [ ] Gitbook: 完整文档编写中...
- [x] Notes: Checkpointer 机制深度分析（存储膨胀问题）

### LangChain
- [ ] 待启动

### Mem0
- [ ] 待启动
