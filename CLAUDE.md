# CLAUDE.md

Agent / LLM 开源框架深度学习仓库。使用 VitePress 构建文档站点。

## 目录结构

每个框架一个顶层目录，内含四个子目录：

- `source/` — 框架源码（git submodule，只读）
- `gitbook/` — 系统化框架解读文档，像一本书。按章节编号 `00-index.md`, `01-xxx.md` ...
- `notes/` — 问题驱动的研究笔记。官方文档没答案的问题、生产踩坑
- `examples/` — 实验代码

gitbook 和 notes 的区别：gitbook 是教科书（系统化、有章节），notes 是实验笔记（问题驱动、深挖某一点）。

## 常用命令

```bash
npm run dev      # 本地预览 http://localhost:5173
npm run build    # 构建静态站点到 .vitepress/dist/
npm run preview  # 预览构建产物
```

## 文档编写约定

- 语言：中文为主，代码和专有名词保持英文
- 图表：使用 Excalidraw 制作，源文件 `.excalidraw` 和导出的 `.png` 都放在对应的 `diagrams/` 下
- gitbook 章节以数字编号开头：`01-overview.md`, `02-architecture.md`
- notes 也以数字编号开头：`01-checkpointer-deep-dive.md`
- 新增章节后需同步更新 `.vitepress/config.mjs` 中的 sidebar 配置
- Markdown 中引用图片使用相对路径：`![title](diagrams/xxx.png)`

## 新增框架的流程

1. `git submodule add <repo-url> <framework>/source`
2. 创建 `<framework>/gitbook/`、`<framework>/notes/`、`<framework>/examples/`
3. 在 `.vitepress/config.mjs` 中添加 nav 和 sidebar 条目
4. 从 `00-index.md` 开始编写 gitbook

## 部署

推送到 master 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。

部署前需在 GitHub 仓库 Settings → Pages 中将 Source 设为 "GitHub Actions"。
