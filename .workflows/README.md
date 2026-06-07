# Framework Gitbook Builder Workflow

自动为开源框架生成 gitbook 式深度解读文档的 workflow。

## 使用方式

在 Claude Code 中执行：

```
workflow build-framework-gitbook
```

或直接引用脚本：

```
Workflow({scriptPath: ".workflows/build-framework-gitbook.js"})
```

## 工作流程

```
Phase 1: Research (并行)
  ├── Agent: 研究框架 A（源码结构 + 网络调研）
  └── Agent: 研究框架 B（源码结构 + 网络调研）
      ↓
Phase 2: Generate (并行)
  ├── Agent: 基于研究结果生成框架 A 的 gitbook（8 章）
  └── Agent: 基于研究结果生成框架 B 的 gitbook（8 章）
      ↓
输出: JSON 对象 {filename: markdown_content}
```

## 定制新框架

修改 `FRAMEWORKS` 数组，添加新框架的配置：

```javascript
{
  name: 'new-framework',        // 目录名
  repo: 'org/repo',             // GitHub 仓库
  sourcePath: '/path/to/source', // 本地源码路径
  description: '...',
  researchPrompt: '...',        // 研究阶段的 prompt
}
```

Generate 阶段的章节结构也需要根据框架特点调整。

## 输出约定

- 文件编号 `00-index.md` 到 `07-production.md`
- 中文为主，代码和技术术语保持英文
- 每章末尾有"下一章"链接
- 包含"设计思考"分析段落
- 生成后需手动更新 `.vitepress/config.mjs` 的 sidebar
