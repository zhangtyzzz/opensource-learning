import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Open Source Learning',
  description: 'Agent / LLM 开源框架深度学习',
  lang: 'zh-CN',
  lastUpdated: true,
  ignoreDeadLinks: true,

  srcExclude: ['**/source/**', '**/examples/**'],

  markdown: {
    theme: { light: 'github-dark', dark: 'github-dark' },
  },

  themeConfig: {
    nav: [
      { text: 'LangGraph', link: '/langgraph/gitbook/00-index' },
      { text: 'LangChain', link: '/langchain/' },
      { text: 'Mem0', link: '/mem0/' },
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
