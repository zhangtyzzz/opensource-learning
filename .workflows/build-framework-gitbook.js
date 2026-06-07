export const meta = {
  name: 'build-framework-gitbook',
  description: 'Research a single framework, plan chapter structure, then generate gitbook docs',
  phases: [
    { title: 'Research', detail: 'Deep research on framework architecture, concepts, and source code' },
    { title: 'Plan', detail: 'Design chapter structure based on research findings' },
    { title: 'Generate', detail: 'Generate each chapter in parallel' },
  ],
}

// args: { name, sourcePath, description? }
// Example: { name: "langchain", sourcePath: "/path/to/langchain/source", description: "LLM application framework" }
const fw = args
if (!fw || !fw.name || !fw.sourcePath) {
  return { error: 'args required: { name: string, sourcePath: string, description?: string }' }
}

// Phase 1: Research
phase('Research')
log('Researching ' + fw.name + '...')

const researchResult = await agent(
  'You are researching the open-source framework "' + fw.name + '" for a deep-dive technical gitbook.\n' +
  (fw.description ? 'Context: ' + fw.description + '\n' : '') +
  '\nDo TWO things:\n\n' +
  '1. READ the source code structure at ' + fw.sourcePath + ':\n' +
  '   - List top-level directories\n' +
  '   - Identify the main library package(s) and their module structure\n' +
  '   - Find key source files for core abstractions\n' +
  '   - Note the dependency structure between packages\n\n' +
  '2. SEARCH THE WEB for:\n' +
  '   - Official documentation and its structure\n' +
  '   - Architecture overviews and design philosophy\n' +
  '   - Core concepts and abstractions\n' +
  '   - Key design decisions and known tradeoffs\n' +
  '   - Community discussions about limitations or pain points\n\n' +
  'Return a comprehensive research report covering:\n' +
  '- What this framework is and what problem it solves\n' +
  '- Where it sits in the ecosystem (competitors, complementary tools)\n' +
  '- Design philosophy and inspirations\n' +
  '- Package/module structure (from source code)\n' +
  '- Core abstractions (list each with description)\n' +
  '- Key design tradeoffs and known limitations\n' +
  '- What the official docs cover well vs what they skip\n' +
  '- Source code paths for key modules\n\n' +
  'Be thorough — this research drives the entire gitbook structure.',
  { label: 'research:' + fw.name, phase: 'Research' }
)

// Phase 2: Plan chapter structure
phase('Plan')
log('Planning gitbook structure for ' + fw.name + '...')

const planResult = await agent(
  'Based on the following research about "' + fw.name + '", design a gitbook chapter structure.\n\n' +
  'RESEARCH:\n' + researchResult + '\n\n' +
  'Your job is to decide:\n' +
  '1. How many chapters are needed (typically 6-12, but let the content decide)\n' +
  '2. What each chapter covers\n' +
  '3. The ordering (should build understanding progressively)\n' +
  '4. Which chapters need architecture diagrams\n\n' +
  'Rules:\n' +
  '- Chapter 00 is always the index/TOC\n' +
  '- Chapter 01 is always the overview (what, why, positioning)\n' +
  '- Chapter 02 is always the architecture (module structure, data flow)\n' +
  '- The last chapter is always production practices (deployment, pitfalls, tradeoffs)\n' +
  '- Middle chapters cover core concepts — one concept per chapter, not too thin, not too thick\n' +
  '- Include a "设计思考" (Design Thinking) section in relevant chapters analyzing WHY\n' +
  '- Every chapter must earn its place — no generic filler\n\n' +
  'Return a JSON object with a "chapters" key containing an array of chapter objects.\n' +
  'Example: {"chapters": [{"number": "00", "slug": "index", "title": "目录", "description": "...", "key_topics": ["topic1"]}, ...]}\n' +
  'You MUST call the StructuredOutput tool with the result.',
  {
    label: 'plan:' + fw.name,
    phase: 'Plan',
    schema: {
      type: 'object',
      properties: {
        chapters: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              number: { type: 'string' },
              slug: { type: 'string' },
              title: { type: 'string' },
              description: { type: 'string' },
              key_topics: { type: 'array', items: { type: 'string' } },
            },
            required: ['number', 'slug', 'title', 'description', 'key_topics'],
          },
        },
      },
      required: ['chapters'],
    },
  }
)

log('Planned ' + planResult.chapters.length + ' chapters for ' + fw.name)
var plan = planResult.chapters

// Phase 3: Generate chapters in parallel
phase('Generate')

var displayName = fw.name.charAt(0).toUpperCase() + fw.name.slice(1)
if (fw.name === 'langchain') displayName = 'LangChain'
if (fw.name === 'langgraph') displayName = 'LangGraph'

// Build index page from plan
var indexRows = ''
var contentChapters = []
for (var ci = 0; ci < plan.length; ci++) {
  var ch = plan[ci]
  if (ch.number !== '00') {
    indexRows += '| [' + ch.number + ' - ' + ch.title + '](' + ch.number + '-' + ch.slug + '.md) | ' + ch.description + ' | |\n'
    contentChapters.push(ch)
  }
}

var indexContent = '# ' + displayName + ' 深度解读\n\n' +
  '> 基于源码和官方文档的系统化学习笔记。不是 API 教程，而是理解"为什么这样设计"。\n\n' +
  '## 目录\n\n' +
  '| 章节 | 内容 | 状态 |\n' +
  '|------|------|------|\n' +
  indexRows + '\n' +
  '## 对应源码\n\n' +
  '源码在 `../source/` 目录下（git submodule）。\n'

// Generate content chapters in parallel
var chapterPrompts = []
for (var j = 0; j < contentChapters.length; j++) {
  var chapter = contentChapters[j]
  var nextLink = ''
  if (j < contentChapters.length - 1) {
    var next = contentChapters[j + 1]
    nextLink = 'End with: > **下一章**: [' + next.number + ' - ' + next.title + '](' + next.number + '-' + next.slug + '.md)\n'
  } else {
    nextLink = 'This is the last chapter. No next link needed.\n'
  }

  var planSummary = ''
  for (var k = 0; k < plan.length; k++) {
    planSummary += plan[k].number + ' - ' + plan[k].title + ': ' + plan[k].description + '\n'
  }

  chapterPrompts.push({
    chapter: chapter,
    prompt: 'Write chapter ' + chapter.number + ' of the ' + fw.name + ' gitbook.\n\n' +
      'CHAPTER INFO:\n' +
      '- Title: ' + chapter.title + '\n' +
      '- Description: ' + chapter.description + '\n' +
      '- Key topics: ' + chapter.key_topics.join(', ') + '\n\n' +
      'FRAMEWORK RESEARCH:\n' + researchResult + '\n\n' +
      'FULL CHAPTER PLAN:\n' + planSummary + '\n\n' +
      'WRITING GUIDELINES:\n' +
      '- Write in Chinese, keep code and technical terms in English\n' +
      '- Start with: # ' + chapter.number + ' - ' + chapter.title + '\n' +
      '- Code examples must have language tags (python, bash, json, sql etc.)\n' +
      '- ASCII diagrams use plain code blocks (no language tag)\n' +
      '- Include "## 设计思考" sections analyzing WHY decisions were made\n' +
      '- ' + nextLink +
      '- Be specific and technical, reference source code paths\n' +
      '- Target: 200-400 lines of markdown\n\n' +
      'Return ONLY the markdown content.',
  })
}

var chapterResults = await parallel(
  chapterPrompts.map(function(cp) {
    return function() {
      return agent(cp.prompt, {
        label: 'write:' + fw.name + ':' + cp.chapter.number,
        phase: 'Generate',
      })
    }
  })
)

// Assemble files
var files = {}
files['00-index.md'] = indexContent
for (var m = 0; m < contentChapters.length; m++) {
  if (chapterResults[m]) {
    files[contentChapters[m].number + '-' + contentChapters[m].slug + '.md'] = chapterResults[m]
  }
}

// Build sidebar config for vitepress
var sidebarItems = []
for (var n = 0; n < plan.length; n++) {
  sidebarItems.push({
    text: plan[n].number + ' ' + plan[n].title,
    link: '/' + fw.name + '/gitbook/' + plan[n].number + '-' + plan[n].slug,
  })
}

return {
  framework: fw.name,
  plan: plan,
  files: files,
  sidebarConfig: sidebarItems,
}
