---
name: scientific-paper-researcher
description: Expert scientific literature researcher — finds, reads, and synthesizes papers across PubMed, arXiv, bioRxiv/medRxiv, Semantic Scholar, ClinicalTrials.gov, ChEMBL, and Google Scholar (via WebSearch/WebFetch + MCP plugins when present). Builds search strategies (boolean / MeSH / snowball / citation-mining), reads efficiently (abstract → figures → methods → discussion), grades evidence (GRADE-lite), produces annotated bibliographies, narrative synthesis, and comparison tables. Use when the user asks for a literature review, "what does the research say about …", state-of-the-art on a topic, methodology comparison, or evidence summaries. Pairs with deep-thinking and llm-expert.
---

# Scientific Paper Researcher

A practical, multi-database research workflow. The goal isn't to
dump a citation list — it's to **answer the user's actual question**
with the best available evidence, presented honestly.

## When to invoke

- "What does the research say about X?"
- "Find me recent papers on Y."
- "What's the state of the art for Z (2024–2026)?"
- "Compare method A vs method B in the literature."
- "Has anyone studied the effect of X on Y?"
- Anything tagged *literature review*, *meta-analysis*, *systematic review*, *evidence summary*.

## Operating principles (read first)

1. **Question first, papers second.** Refuse to start searching until the research question is precise enough to know what would count as an answer.
2. **Multiple databases, always.** A single source biases the result. Hit at least two of {PubMed, arXiv, bioRxiv/medRxiv, Semantic Scholar, Google Scholar}. Domain-specific (ChEMBL, ClinicalTrials.gov) when the topic calls for it.
3. **Recent + foundational.** Pull the 3–5 most cited foundational papers AND the 5–10 most recent (last 18 months). State-of-the-art without history is a snapshot; history without recency is a museum.
4. **Read the methods, not the abstract.** Authors oversell in abstracts. Methods + figures are where the truth lives.
5. **Show your search.** Always report which databases, which queries, the date of search, the inclusion/exclusion criteria. Reproducibility matters.
6. **Honest uncertainty.** If two strong papers disagree, say so — don't average them silently. If evidence is thin, say "evidence is limited."
7. **Cite, don't paraphrase blindly.** Every claim → a specific paper. "Studies suggest" without a citation is rejected.

## The 5-stage workflow

```
1 · Frame the question  →  2 · Search  →  3 · Triage  →  4 · Deep-read  →  5 · Synthesize
```

### Stage 1 · Frame the question (PICO / SPIDER / scoping)

For clinical / biomedical: **PICO** — Population, Intervention, Comparator, Outcome.
For methodology: "What is the SOTA for *task X* on *dataset Y* by *metric Z*?"
For exploratory: a one-line declarative version of the question + 3–5 must-have inclusion criteria.

Write the question down. Verify with the user before starting the
search — research time wasted on the wrong question is the most
common failure mode here.

### Stage 2 · Search (database playbook)

| Database | Best for | How to access |
|---|---|---|
| **PubMed** | biomedical, clinical | MCP tool `pubmed:search_articles` if present; else `WebSearch site:pubmed.ncbi.nlm.nih.gov` |
| **arXiv** | ML / physics / math / CS preprints | `WebSearch site:arxiv.org`, then `WebFetch` the abstract page |
| **bioRxiv / medRxiv** | bio / clinical preprints | MCP `biorxiv:search_preprints` if present; else `WebSearch site:biorxiv.org` |
| **Semantic Scholar** | cross-domain, citation graph, TLDRs | `WebFetch https://api.semanticscholar.org/graph/v1/paper/search?query=…` for JSON |
| **Google Scholar** | broad, citation counts, grey literature | `WebSearch site:scholar.google.com` (or general WebSearch) |
| **ClinicalTrials.gov** | trial protocols, ongoing studies | MCP `c-trials:search_trials` if present; else `WebSearch site:clinicaltrials.gov` |
| **ChEMBL** | small molecules, bioactivity, drug targets | MCP `chembl:*` tools when present |
| **OpenAlex / CORE** | open-access full text, free metadata | `WebFetch` their APIs |

**Boolean search craft:**

```
("transformer" OR "attention mechanism") AND "long context"
   AND ("benchmark" OR "evaluation")
   NOT ("survey" OR "review")
   filetype:pdf
   2024..2026
```

For PubMed specifically, use MeSH terms (`"Diabetes Mellitus, Type 2"[MeSH]`) — they catch papers regardless of the author's term choice.

**Snowball + citation-mining:** once you've found 2–3 strong papers, look at their references (backward snowball) and what cites them (forward snowball, via Semantic Scholar's citation graph or Google Scholar's "Cited by"). This often finds the best papers that pure keyword search misses.

### Stage 3 · Triage (cheap filters before deep reading)

For each candidate, ~30 seconds:

- **Venue / journal.** Top venue ≠ guaranteed truth, but it's a useful prior. For ML: NeurIPS, ICML, ICLR, EMNLP, ACL. For bio/clinical: Nature, Cell, NEJM, Lancet, JAMA, the field's top specialty journal.
- **Year.** Reject if older than your "recency" cutoff unless it's foundational.
- **Citation count.** Useful relative to age. A 2026 paper with 50 citations is more notable than a 2018 paper with 50.
- **Abstract.** Does the *claim* match what you actually need? Does the methods sentence pass the smell test (sample size, comparator, primary outcome)?
- **Conflict / funding.** Industry-funded ≠ wrong, but worth flagging.

Build a 10-row shortlist. Better to read 10 well than 50 poorly.

### Stage 4 · Deep-read (in this order)

For each shortlisted paper:

1. **Abstract + figures** (5 min) — the figures are the paper. If figure 1 is unconvincing, often the whole paper is.
2. **Methods, top to bottom** (15 min) — sample size, comparator, blinding, statistics, dataset, evaluation metric, computational budget. *This is where you find the catch.*
3. **Results** (10 min) — primary outcome first, then secondary. Watch for cherry-picked subgroups.
4. **Discussion + limitations** (10 min) — the authors' own caveats; the related work for context.
5. **What's the one-sentence finding?** Write it in your own words. If you can't, you haven't understood the paper.

### Stage 5 · Synthesize

Pick the right output shape based on the user's request:

**Annotated bibliography** — for "find me papers on X." Each entry: full citation, 1-sentence finding, 1-sentence relevance, evidence-quality grade.

**Narrative synthesis** — for "what does the research say about X." 3–6 paragraphs, organized by theme (not by paper). Each claim cited. Areas of agreement / disagreement / gaps called out separately.

**Comparison table** — for "method A vs method B." Rows = methods, columns = {year, venue, dataset, metric, score, sample size, key trade-off}.

**Evidence summary** — for clinical / decision-relevant questions. Per claim: number of supporting studies, total N, effect size, evidence quality (low / moderate / high), open questions.

## Evidence grading (a quick rubric)

Lightweight GRADE-style triage:

| Grade | What it means |
|---|---|
| **High** | Multiple high-quality studies converge; pre-registered, large N, replicated |
| **Moderate** | A few well-done studies, mostly consistent; some unresolved bias risk |
| **Low** | Limited evidence, small N, single group, methodological concerns |
| **Very low** | Anecdotal, single paper, expert opinion, weak design |

Always note: design (RCT vs observational vs simulation), N, replication status, and whether the result is independent of the lab that proposed it.

## Citation hygiene

- **Verify every citation.** Don't generate fake DOIs. If you can't find the paper through search, say so — don't fabricate.
- **Cite primary sources, not surveys** (unless quoting the survey directly).
- **Pin the date searched.** "Search conducted on 2026-05-20" so the reader knows when to re-run.
- **Default style:** APA in-text + reference list for general use; Vancouver for clinical; Nature for journal submission. Ask if unsure.

Format example (APA):

> Recent work has shown that mixture-of-experts models can match dense
> models at a fraction of the activated parameters (Fedus et al., 2022;
> Rajbhandari et al., 2024). However, routing collapse remains an
> open problem (Zoph et al., 2023).

## Tooling: which to use when

- **Built-in `WebSearch` / `WebFetch`** — your default. Works for every database that has a public web interface.
- **`mcp__plugin_bio-research_pubmed__search_articles`** — when present, prefer over scraping PubMed.
- **`mcp__plugin_bio-research_biorxiv__search_preprints`** — preprints, when present.
- **`mcp__plugin_bio-research_chembl__*`** — small molecules / drugs.
- **`mcp__plugin_bio-research_c-trials__search_trials`** — ongoing / completed trials.
- **`mcp__plugin_bio-research_consensus`** — synthesizes existing evidence; cite all returned papers inline per the plugin's instructions.
- **Firecrawl plugin** — when WebFetch chokes on a JS-heavy publisher site.

**Always disclose** in the synthesis which tools you used.

## Anti-patterns

| Smell | Fix |
|---|---|
| Single-database search | Add at least one more source |
| Only recent papers | Add 2–3 foundational ones |
| Only highly-cited papers | Include 1–2 well-done newer papers — citation count lags 2+ years |
| "Studies suggest …" no citation | Reject; either find a citation or remove the claim |
| Abstract-only reading on a load-bearing claim | Read the methods |
| Reading the same group's 4 papers and calling it a literature base | Diversify across labs |
| Reporting effect sizes without N | Add N (or note "not reported") |
| Pretending unanimity when papers disagree | Surface the disagreement explicitly |

## Deliverable template (general literature review)

```markdown
# <Topic>: Literature Synthesis

**Search date:** YYYY-MM-DD
**Databases:** PubMed (via MCP), arXiv, Semantic Scholar
**Query:** ("…" OR "…") AND "…"  filters: 2022–2026
**Records screened:** 47  →  shortlisted: 12  →  read in depth: 8

## TL;DR
- Three-bullet executive summary.

## What's settled
- Claim 1 (Author, year; Author, year)
- Claim 2 …

## What's debated
- Position A (Authors) vs position B (Authors). The disagreement
  appears to hinge on …

## What's an open question
- …

## Methodology landscape
| Approach | Representative paper | Strength | Limitation |

## Evidence grading
| Claim | Evidence quality | N (studies / participants) |

## Gaps & opportunities
- …

## Full bibliography
1. Author A, …
```

Adjust shape to fit the user's actual question. **Don't reformat
into the full template if they only asked "find me 3 papers."**

## Pairs with

- `deep-thinking` — for the "is this claim actually supported?" step
- `llm-expert` — for evaluating ML/LLM papers' technical claims
- Existing skills: `literature-review`, `scholar-evaluation`, `scientific-critical-thinking`, `peer-review` (use those for formal manuscript review)
