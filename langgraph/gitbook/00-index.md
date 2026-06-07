# LangGraph 深度解读

> 基于 LangGraph 源码和官方文档的系统化学习笔记。不是 API 教程，而是理解"为什么这样设计"。

## 目录

| 章节 | 内容 | 状态 |
|------|------|------|
| [01 - 框架概览](01-overview.md) | 是什么、为什么、和 LangChain 的关系 | |
| [02 - 架构总览](02-architecture.md) | 分层架构、代码仓库结构、核心模块 | |
| [03 - 核心概念](03-core-concepts.md) | Graph / State / Node / Edge / Reducer | |
| [04 - Channel 机制](04-channels.md) | 状态通信的核心抽象 | |
| [05 - Pregel 引擎](05-pregel-engine.md) | BSP 执行模型、SuperStep、任务调度 | |
| [06 - 持久化与 Checkpointer](06-checkpointer.md) | 检查点、Store、序列化 | |
| [07 - Streaming 机制](07-streaming.md) | 7 种流模式、v2 格式 | |
| [08 - Human-in-the-Loop](08-human-in-the-loop.md) | interrupt / Command / 状态修改 | |
| [09 - 工作流模式](09-workflow-patterns.md) | 6 种编排模式及选型 | |
| [10 - 生产实践](10-production.md) | 性能、部署、踩坑 | |

## 配图

所有架构图使用 Excalidraw 制作，源文件在 [diagrams/](diagrams/) 目录下。

## 对应源码

源码在 `../source/` 目录下（git submodule），核心代码路径：

```
source/libs/langgraph/langgraph/     # 核心库
  ├── graph/                          # Graph 构建 API
  ├── pregel/                         # 执行引擎（最大最复杂的模块）
  ├── channels/                       # 状态通道
  ├── stream/                         # 流式输出
  ├── func/                           # Functional API (@task, @entrypoint)
  └── managed/                        # 可注入的运行时值

source/libs/checkpoint/               # Checkpointer 基础库
source/libs/checkpoint-postgres/      # PostgreSQL 实现
source/libs/prebuilt/                 # 预构建组件 (create_react_agent 等)
```
