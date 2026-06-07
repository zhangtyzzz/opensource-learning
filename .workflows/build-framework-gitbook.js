export const meta = {
  name: 'build-framework-gitbook',
  description: 'Research, plan, generate, and quality-check gitbook docs for a framework',
  phases: [
    { title: 'Research', detail: 'Deep research on framework architecture, concepts, and source code' },
    { title: 'Plan', detail: 'Design chapter structure based on research findings' },
    { title: 'Generate', detail: 'Generate each chapter in parallel' },
    { title: 'Judge', detail: 'Quality check each chapter, flag failures' },
    { title: 'Fix', detail: 'Rewrite failed chapters and re-judge (up to 3 rounds)' },
  ],
}

// args: { name, sourcePath, description? }
var fw = args
if (!fw || !fw.name || !fw.sourcePath) {
  return { error: 'args required: { name: string, sourcePath: string, description?: string }' }
}

var MAX_FIX_ROUNDS = 3

// ─── Phase 1: Research ───
phase('Research')
log('Researching ' + fw.name + '...')

var researchResult = await agent(
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

// ─── Phase 2: Plan ───
phase('Plan')
log('Planning gitbook structure for ' + fw.name + '...')

var planResult = await agent(
  'Based on the following research about "' + fw.name + '", design a gitbook chapter structure.\n\n' +
  'RESEARCH:\n' + researchResult + '\n\n' +
  'Your job is to decide:\n' +
  '1. How many chapters are needed (typically 6-12, but let the content decide)\n' +
  '2. What each chapter covers\n' +
  '3. The ordering (should build understanding progressively)\n\n' +
  'Rules:\n' +
  '- Chapter 00 is always the index/TOC\n' +
  '- Chapter 01 is always the overview (what, why, positioning)\n' +
  '- Chapter 02 is always the architecture (module structure, data flow)\n' +
  '- The last chapter is always production practices (deployment, pitfalls, tradeoffs)\n' +
  '- Middle chapters cover core concepts — one concept per chapter\n' +
  '- Every chapter must earn its place — no generic filler\n\n' +
  'Return a JSON object with a "chapters" key containing an array of chapter objects.\n' +
  'Example: {"chapters": [{"number": "00", "slug": "index", "title": "...", "description": "...", "key_topics": ["topic1"]}, ...]}\n' +
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

// ─── Phase 3: Generate ───
phase('Generate')

var displayName = fw.name.charAt(0).toUpperCase() + fw.name.slice(1)
if (fw.name === 'langchain') displayName = 'LangChain'
if (fw.name === 'langgraph') displayName = 'LangGraph'

// Separate index from content chapters
var contentChapters = []
for (var ci = 0; ci < plan.length; ci++) {
  if (plan[ci].number !== '00') {
    contentChapters.push(plan[ci])
  }
}

// Build index page
var indexRows = ''
for (var ir = 0; ir < contentChapters.length; ir++) {
  indexRows += '| [' + contentChapters[ir].number + ' - ' + contentChapters[ir].title + '](' + contentChapters[ir].number + '-' + contentChapters[ir].slug + '.md) | ' + contentChapters[ir].description + ' | |\n'
}
var indexContent = '# ' + displayName + ' 深度解读\n\n' +
  '> 基于源码和官方文档的系统化学习笔记。不是 API 教程，而是理解"为什么这样设计"。\n\n' +
  '## 目录\n\n| 章节 | 内容 | 状态 |\n|------|------|------|\n' + indexRows +
  '\n## 对应源码\n\n源码在 `../source/` 目录下（git submodule）。\n'

// Build plan summary for context
var planSummary = ''
for (var ps = 0; ps < plan.length; ps++) {
  planSummary += plan[ps].number + ' - ' + plan[ps].title + ': ' + plan[ps].description + '\n'
}

// Helper: build generation prompt for a chapter
function buildWritePrompt(chapter, chapterIndex) {
  var nextLink = ''
  if (chapterIndex < contentChapters.length - 1) {
    var next = contentChapters[chapterIndex + 1]
    nextLink = 'End with: > **下一章**: [' + next.number + ' - ' + next.title + '](' + next.number + '-' + next.slug + '.md)\n'
  } else {
    nextLink = 'This is the last chapter. No next link needed.\n'
  }

  return 'Write chapter ' + chapter.number + ' of the ' + fw.name + ' gitbook.\n\n' +
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
    '- Target: 200-400 lines of markdown\n' +
    '- Do NOT use bare {{ }} in text (VitePress treats them as Vue templates)\n\n' +
    'CRITICAL: Return ONLY the raw markdown content of the chapter.\n' +
    'Do NOT say "I have written the file" or "Chapter written to...".\n' +
    'Do NOT use the Write tool. Just return the markdown text directly as your final response.'
}

// Generate all chapters in parallel
var chapterResults = await parallel(
  contentChapters.map(function(ch, idx) {
    return function() {
      return agent(buildWritePrompt(ch, idx), {
        label: 'write:' + fw.name + ':' + ch.number,
        phase: 'Generate',
      })
    }
  })
)

// ─── Phase 4: Judge ───
phase('Judge')
log('Judging quality of ' + chapterResults.length + ' chapters...')

// Helper: judge a single chapter
function buildJudgePrompt(chapter, content) {
  return 'You are a strict quality judge for technical documentation.\n\n' +
    'CHAPTER: ' + chapter.number + ' - ' + chapter.title + '\n' +
    'EXPECTED TOPICS: ' + chapter.key_topics.join(', ') + '\n\n' +
    'CONTENT TO JUDGE:\n' + content + '\n\n' +
    'Check ALL of the following criteria:\n\n' +
    '1. LENGTH: Content must be at least 100 lines of actual markdown (not counting blank lines). Count the lines.\n' +
    '2. NOT META: Content must be actual chapter text, NOT a meta-description like "Chapter written to file" or "I have generated...".\n' +
    '3. STRUCTURE: Must start with a # heading, must have at least 3 ## subheadings.\n' +
    '4. CODE EXAMPLES: Must contain at least 1 code block with a language tag (```python etc).\n' +
    '5. TECHNICAL DEPTH: Must contain specific technical details (class names, method signatures, source paths), not just generic descriptions.\n' +
    '6. ACCURACY: No obviously wrong technical claims. Check that class names, method names, and described behaviors are plausible for this framework.\n' +
    '7. COMPLETENESS: All key_topics listed above should be addressed, not just mentioned in passing.\n' +
    '8. VITEPRESS SAFE: Must not contain bare {{ }} outside of code blocks (Vue template conflict).\n\n' +
    'Return your verdict as JSON:\n' +
    '{"pass": true/false, "failures": ["criterion 1 description", ...], "suggestions": "one paragraph of what to fix"}\n\n' +
    'Be strict. If any criterion fails, set pass=false. A chapter that is just a placeholder or meta-text should fail immediately.'
}

// Judge all chapters in parallel
var judgeResults = await parallel(
  contentChapters.map(function(ch, idx) {
    var content = chapterResults[idx] || ''
    return function() {
      return agent(buildJudgePrompt(ch, content), {
        label: 'judge:' + fw.name + ':' + ch.number,
        phase: 'Judge',
        schema: {
          type: 'object',
          properties: {
            pass: { type: 'boolean' },
            failures: { type: 'array', items: { type: 'string' } },
            suggestions: { type: 'string' },
          },
          required: ['pass', 'failures', 'suggestions'],
        },
      })
    }
  })
)

// Count failures
var failedIndices = []
for (var ji = 0; ji < judgeResults.length; ji++) {
  if (judgeResults[ji] && !judgeResults[ji].pass) {
    failedIndices.push(ji)
    log('FAIL: Chapter ' + contentChapters[ji].number + ' - ' + judgeResults[ji].failures.join('; '))
  }
}
log(failedIndices.length + ' of ' + contentChapters.length + ' chapters failed quality check')

// ─── Phase 5: Fix loop (up to MAX_FIX_ROUNDS) ───
var round = 0
while (failedIndices.length > 0 && round < MAX_FIX_ROUNDS) {
  round++
  phase('Fix')
  log('Fix round ' + round + '/' + MAX_FIX_ROUNDS + ': rewriting ' + failedIndices.length + ' chapters...')

  // Rewrite failed chapters in parallel
  var fixResults = await parallel(
    failedIndices.map(function(fi) {
      var ch = contentChapters[fi]
      var judgment = judgeResults[fi]
      var previousContent = chapterResults[fi] || ''
      return function() {
        return agent(
          'Your previous draft of chapter ' + ch.number + ' FAILED quality review.\n\n' +
          'FAILURES:\n' + judgment.failures.join('\n') + '\n\n' +
          'REVIEWER SUGGESTIONS:\n' + judgment.suggestions + '\n\n' +
          'PREVIOUS DRAFT (may be empty or broken):\n' +
          (previousContent.length > 200 ? previousContent.substring(0, 200) + '\n... (truncated)' : previousContent) + '\n\n' +
          'Now rewrite the chapter from scratch.\n\n' +
          buildWritePrompt(ch, fi),
          {
            label: 'fix:' + fw.name + ':' + ch.number + ':r' + round,
            phase: 'Fix',
          }
        )
      }
    })
  )

  // Update chapter results
  for (var fi2 = 0; fi2 < failedIndices.length; fi2++) {
    chapterResults[failedIndices[fi2]] = fixResults[fi2]
  }

  // Re-judge fixed chapters
  log('Re-judging ' + failedIndices.length + ' fixed chapters...')
  var reJudgeResults = await parallel(
    failedIndices.map(function(fi) {
      var ch = contentChapters[fi]
      var content = chapterResults[fi] || ''
      return function() {
        return agent(buildJudgePrompt(ch, content), {
          label: 'rejudge:' + fw.name + ':' + ch.number + ':r' + round,
          phase: 'Fix',
          schema: {
            type: 'object',
            properties: {
              pass: { type: 'boolean' },
              failures: { type: 'array', items: { type: 'string' } },
              suggestions: { type: 'string' },
            },
            required: ['pass', 'failures', 'suggestions'],
          },
        })
      }
    })
  )

  // Update judge results and find remaining failures
  var newFailedIndices = []
  for (var rj = 0; rj < failedIndices.length; rj++) {
    var origIdx = failedIndices[rj]
    judgeResults[origIdx] = reJudgeResults[rj]
    if (reJudgeResults[rj] && !reJudgeResults[rj].pass) {
      newFailedIndices.push(origIdx)
      log('Still failing: Chapter ' + contentChapters[origIdx].number + ' - ' + reJudgeResults[rj].failures.join('; '))
    } else {
      log('Fixed: Chapter ' + contentChapters[origIdx].number)
    }
  }
  failedIndices = newFailedIndices
}

if (failedIndices.length > 0) {
  log('WARNING: ' + failedIndices.length + ' chapters still failing after ' + MAX_FIX_ROUNDS + ' fix rounds')
}

// ─── Assemble final output ───
var files = {}
files['00-index.md'] = indexContent
for (var m = 0; m < contentChapters.length; m++) {
  if (chapterResults[m]) {
    files[contentChapters[m].number + '-' + contentChapters[m].slug + '.md'] = chapterResults[m]
  }
}

var sidebarItems = []
for (var n = 0; n < plan.length; n++) {
  sidebarItems.push({
    text: plan[n].number + ' ' + plan[n].title,
    link: '/' + fw.name + '/gitbook/' + plan[n].number + '-' + plan[n].slug,
  })
}

// Build quality report
var qualityReport = []
for (var qr = 0; qr < contentChapters.length; qr++) {
  qualityReport.push({
    chapter: contentChapters[qr].number + ' ' + contentChapters[qr].title,
    pass: judgeResults[qr] ? judgeResults[qr].pass : false,
    failures: judgeResults[qr] ? judgeResults[qr].failures : ['no judge result'],
  })
}

return {
  framework: fw.name,
  plan: plan,
  files: files,
  sidebarConfig: sidebarItems,
  qualityReport: qualityReport,
  fixRoundsUsed: round,
  remainingFailures: failedIndices.length,
}
