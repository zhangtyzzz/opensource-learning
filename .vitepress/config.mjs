import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Open Source Learning',
  description: 'Agent / LLM 开源框架深度学习',
  lang: 'zh-CN',
  base: '/opensource-learning/',
  lastUpdated: true,
  ignoreDeadLinks: true,

  srcExclude: ['**/source/**', '**/examples/**'],

  markdown: {
    theme: { light: 'github-dark', dark: 'github-dark' },
  },

  themeConfig: {
    nav: [
      { text: 'LangGraph', link: '/langgraph/gitbook/00-index' },
      { text: 'LangChain', link: '/langchain/gitbook/00-index' },
      { text: 'Mem0', link: '/mem0/gitbook/00-index' },
    ],

    sidebar: {
      '/langgraph/': [
        {
          text: 'Gitbook — 框架解读',
          items: [
            { text: '目录', link: '/langgraph/gitbook/00-index' },
            { text: '01 框架概览', link: '/langgraph/gitbook/01-overview' },
            { text: '02 架构总览', link: '/langgraph/gitbook/02-architecture' },
            { text: '03 核心概念', link: '/langgraph/gitbook/03-core-concepts' },
            { text: '04 Channel 机制', link: '/langgraph/gitbook/04-channels' },
            { text: '05 Pregel 引擎', link: '/langgraph/gitbook/05-pregel-engine' },
            { text: '06 持久化', link: '/langgraph/gitbook/06-checkpointer' },
            { text: '07 Streaming', link: '/langgraph/gitbook/07-streaming' },
            { text: '08 Human-in-the-Loop', link: '/langgraph/gitbook/08-human-in-the-loop' },
            { text: '09 工作流模式', link: '/langgraph/gitbook/09-workflow-patterns' },
            { text: '10 生产实践', link: '/langgraph/gitbook/10-production' },
          ],
        },
        {
          text: 'Notes — 问题研究',
          items: [
            { text: 'Checkpointer 存储膨胀', link: '/langgraph/notes/01-checkpointer-deep-dive' },
          ],
        },
      ],
      '/langchain/': [
        {
          text: 'Gitbook — 框架解读',
          items: [
            { text: '目录', link: '/langchain/gitbook/00-index' },
            { text: '01 全景概览', link: '/langchain/gitbook/01-overview' },
            { text: '02 架构与包设计', link: '/langchain/gitbook/02-architecture' },
            { text: '03 Runnable 协议', link: '/langchain/gitbook/03-runnable-protocol' },
            { text: '04 模型与消息', link: '/langchain/gitbook/04-models-and-messages' },
            { text: '05 Prompt 与输出解析', link: '/langchain/gitbook/05-prompts-and-output-parsing' },
            { text: '06 工具与函数调用', link: '/langchain/gitbook/06-tools-and-tool-calling' },
            { text: '07 检索增强生成', link: '/langchain/gitbook/07-retrieval-and-rag' },
            { text: '08 Agent 与编排', link: '/langchain/gitbook/08-agents-and-orchestration' },
            { text: '09 可观测性', link: '/langchain/gitbook/09-observability-and-tracing' },
            { text: '10 生产实践', link: '/langchain/gitbook/10-production-practices' },
          ],
        },
      ],
      '/mem0/': [
        {
          text: 'Gitbook — 框架解读',
          items: [
            { text: '目录', link: '/mem0/gitbook/00-index' },
            { text: '01 全景概览', link: '/mem0/gitbook/01-overview' },
            { text: '02 架构总览', link: '/mem0/gitbook/02-architecture' },
            { text: '03 核心概念', link: '/mem0/gitbook/03-core-concepts' },
            { text: '04 Provider 插件体系', link: '/mem0/gitbook/04-provider-pattern' },
            { text: '05 写入路径剖析', link: '/mem0/gitbook/05-write-path' },
            { text: '06 读取路径剖析', link: '/mem0/gitbook/06-read-path' },
            { text: '07 图记忆与平台', link: '/mem0/gitbook/07-graph-memory-and-platform' },
            { text: '08 集成生态', link: '/mem0/gitbook/08-integrations' },
            { text: '09 生产实践', link: '/mem0/gitbook/09-production-practices' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/' },
    ],

    search: {
      provider: 'local',
    },

    outline: {
      level: [2, 3],
      label: '目录',
    },

    lastUpdated: {
      text: '最后更新',
    },
  },
})
