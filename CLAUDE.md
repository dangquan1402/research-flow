# Research Flow

An agentic research system with persistent working memory. Multiple agents can work in parallel on the same research goal from different angles, with findings compounding in a shared wiki-style memory.

## Architecture

Three layers (from Karpathy's LLM Wiki pattern):
1. **Sources** (`sources/`) — immutable raw material. Never modify after ingest.
2. **Working Memory** (`memory/`) — LLM-maintained markdown pages. Entities, findings, themes, open questions. Updated every cycle.
3. **Schema** (this file) — conventions, structure, workflows.

Plus a **validation & output layer**:
4. **Outputs** (`outputs/`) — deliverables, evidence, verification, examples, critiques.

```
outputs/
├── evidence/       # Charts, tables, plots backing claims (PNG, HTML, .py scripts)
├── verification/   # Code that checks claims programmatically (.py scripts, results.json)
├── examples/       # Worked examples illustrating findings (markdown, .py)
├── critiques/      # Critical reviews with ratings and gap analysis
└── *.md            # Final deliverables (reports, specs)
```

## Git Flow for Research

| Branch Type | Base | Prefix | Purpose |
|---|---|---|---|
| `research/` | `main` | `research(scope):` | New research goal — creates parent GH issue |
| `hypothesis/` | `research/*` | `hypothesis(scope):` | One angle/approach on a research goal — sub-issue |
| `synthesis/` | `research/*` | `synthesis(scope):` | Merge findings from multiple hypothesis branches |
| `review/` | `main` | `review(scope):` | Lint, reorganize, resolve contradictions in memory |

**Multi-agent parallel work:** Spin up agents on separate `hypothesis/` branches off the same `research/` branch. Each agent explores a different angle. `synthesis/` branches merge the best findings.

**Branch naming:** `{type}/GH-{issue}-{slug}`
Example: `hypothesis/GH-7-attention-mechanisms`, `synthesis/GH-5-consolidate-transformer-findings`

## GitHub Project Tracking

- **Project:** `Research Flow` (project #4, owner: dangquan1402)
- **Repo:** `dangquan1402/research-flow`
- Each research goal = parent issue (label: `research-goal`)
- Each hypothesis/approach = sub-issue under the parent (label: `hypothesis`)
- Each significant finding = sub-issue (label: `finding`)
- Synthesis tasks get label: `synthesis`
- Memory maintenance gets label: `maintenance`
- Project board columns: `Backlog | In Progress | Synthesizing | Done`
- Link all branches to their issues
- Use `gh project item-add 4 --owner dangquan1402 --url {issue_url}` to add issues to project

### Multi-Agent Dispatch

To spin up parallel hypothesis agents:
```
Agent({
  isolation: "worktree",
  prompt: "Research goal: {goal}. Your hypothesis angle: {angle}. Branch: hypothesis/GH-{id}-{slug}. Work in memory/ following CLAUDE.md conventions. When done, commit and push.",
  description: "Hypothesis: {angle}"
})
```
Each agent gets its own worktree (isolated branch). No conflicts during parallel work.

## Memory Structure

```
memory/
├── index.md          # Catalog of all pages with summaries (updated every ingest)
├── log.md            # Append-only chronological record of all operations
├── entities/         # People, orgs, concepts, tools discovered
├── findings/         # Discrete insights with evidence + citations
├── themes/           # Cross-cutting patterns across findings
└── open-questions/   # Gaps identified, queued for next cycle
```

### Memory Rules
- Every memory page has YAML frontmatter: `title`, `created`, `updated`, `source`, `confidence` (high/medium/low), `tags`
- `index.md` is updated after every ingest/analyze/synthesize operation
- `log.md` is append-only with format: `## [YYYY-MM-DD] operation | subject`
- Findings reference their source with `[source-slug]` citations
- Themes cross-reference findings they synthesize
- Open questions link to the research goal they serve

### Memory Hygiene
- No orphan pages (everything in index.md)
- No duplicate entities (check before creating)
- Stale findings (>30 days without re-validation) get flagged
- Contradictions between findings must be surfaced in open-questions/

## Research Loop

1. **Init** (`/research`) — Define goal → create issue → create branch → scaffold
2. **Ingest** (`/analyze`) — Feed source → extract entities/findings → update memory → log
3. **Analyze** (`/analyze`) — Read goal + memory → identify gaps → generate new analysis → update memory
4. **Synthesize** (`/synthesize`) — Consolidate findings → build themes → resolve contradictions → produce output
5. **Evidence** (`/evidence`) — Generate charts, tables, plots backing key claims → `outputs/evidence/`
6. **Verify** (`/verify`) — Generate code that checks intermediate & final claims → `outputs/verification/`
7. **Examples** (`/examples`) — Generate worked examples illustrating findings → `outputs/examples/`
8. **Critique** (`/critique`) — Adversarial review of results, gap analysis, honest write-up → `outputs/critiques/`
9. **Lint** (`/lint`) — Health-check memory: orphans, contradictions, staleness, missing cross-refs

Steps 5-8 form the **validation & presentation layer** — they can be run in any order after synthesis, and each strengthens the others (e.g., verification failures inform critique, examples clarify evidence).

## Quality Patterns

### Verification Gates
Every finding is classified:
- `source` — directly cited from primary material
- `analysis` — inference with reasoning chain shown
- `unverified` — noted but not validated
- `gap` — explicitly missing information

### FUNGI Counter-Arguments
Every finding and theme MUST answer: "What would disprove this?"
This resists confirmation bias and forces rigorous thinking.

### Staleness Scoring
Pages track `staleness_days` in frontmatter. `/lint` increments this. Pages >30 days stale get flagged.

### Entity Registry
`memory/entity-registry.json` prevents duplicate entity pages. Always check before creating.

## Conventions

- Commit messages: `{type}({scope}): description` (e.g., `research(transformers): ingest attention paper`)
- One finding per file in `findings/`
- One entity per file in `entities/`
- Themes can reference multiple findings
- Sources are immutable after creation (enforced by hook)
- Always run `pre-commit` and `ruff` after coding changes
- Templates for all memory pages in `docs/memory-page-template.md`
