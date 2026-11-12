# Engineering notes: making a small model answer like a big one

These notes cover the whole project: the question, the setup, every version we tried, what worked and what failed, the numbers, and the limits of what the numbers show. They are written as the source material for a blog post.

**Short version.** A 4-billion-parameter model (Gemma 4 E4B), driven by code that plans the steps and checks every answer, scored **72%** on 200 hard multi-hop questions. On the 31 questions where we also have answers from a 120-billion-parameter model running its own agent loop (gpt-oss-120b), the small model got **23 right and the big one 24**. We did not change any model weights. The gains came from the pipeline around the model: hybrid search, a reranker, step-by-step code control and answer checks. That pipeline was designed by iterating with a large AI model (Claude, used through Claude Code) that read the small model's mistakes and proposed general fixes.

---

## 1. The question

Agentic RAG lets a large model decide when and what to search, in a loop. It works well but needs a big, expensive model on every query.

**Our hypothesis:** a small model can approach a big agent if *code* does the parts small models are bad at (planning the steps, deciding when to search again, checking answers) and the model only does small reading tasks.

We ran a "race" between several setups on the same questions, the same documents and the same search tools.

---

## 2. Data

### Documents
- **711 new English Wikipedia articles** created between 9 Oct and 7 Nov 2026, plus the articles they link to: 4,102 documents in total (`data/corpus_manifest.json`).
- Because the articles are newer than the models' training data, the models cannot answer from memory; they must search.
- Documents are cut into **140-word chunks with 30-word overlap: 64,262 chunks**. Each chunk is prefixed with its title and section.

### Questions (`data/questions.jsonl`)
200 multi-hop questions, built by `data_build/make_questions.py`:

| Type | Count | What it needs |
|---|---|---|
| `bridge` | 110 | 2 articles: A mentions B; the question describes B only through A and asks a fact from B |
| `bridge3` | 50 | 3 articles in a chain A → B → C |
| `comparison` | 40 | 2 articles; "Who was born earlier, X or Y?" or a yes/no form |

How they were made:
1. Sample linked article pairs or chains from the link graph (seeded, deterministic).
2. An LLM writes a question and short answer from excerpts (Groq: `qwen3.8-27b` and `gpt-oss-20b`). Comparisons use a template.
3. Deterministic checks:
   - the answer is copied from the last-hop excerpt
   - the bridge entity is not named in the question
   - the answer is absent from the first excerpt
4. **Closed-book filter:** if a generator model answers correctly without documents, the question is dropped.
5. Stratified final set, with a 60-question `core60` subset for the expensive racers.

As a sanity check, the small model with **no search** (closed book) scores **2.5%**, so the answers are not in its memory.

---

## 3. The search stack (shared by every racer)

Every racer uses the same tools (`rag/tools.py`, `rag/lance_tools.py`):

| Tool | What it does |
|---|---|
| `grep` | Exact phrase match (names, titles, numbers) |
| `bm25` | Keyword search (stemming, stop-word removal) |
| `vector` | Meaning search with `bge-small-en-v1.5` embeddings (384-d, cosine) |
| `hybrid` | BM25 + vector + exact phrases for named terms, fused with Reciprocal Rank Fusion (RRF); the top 30 are then re-scored by a cross-encoder reranker (`bge-reranker-base`) and the top 5 returned |

We settled on `hybrid` as the main search: in the agent runs it was the best first tool (49% eventual accuracy when the agent started with it, against 40% for BM25).

### One database instead of four structures
The first version kept a BM25 index (`bm25s`), a numpy embedding matrix and a Python grep loop. We moved everything into **one LanceDB table** holding the text, a full-text index with word positions (for exact phrases) and the vectors. One hybrid query does keyword + vector with RRF inside the database.

| Check | Result |
|---|---|
| Vector top-10 overlap with the old tools | 100% |
| BM25 top-10 overlap | 82% |
| Hybrid top-5 overlap | 90% |
| C1 accuracy, old tools vs LanceDB | 62.5% vs 61.0% (no real difference) |

### Is all this searching heavy on the database?
No. Per question (4B model, C1 v3): **2.7 searches, 1.9 s of search time, 96 s of LLM time**. Search is about 2% of the time, and most of it is the cross-encoder reranker running on CPU. The lookups themselves use indexes and take milliseconds to a few hundred milliseconds at 64K chunks. At larger scale you would add an ANN vector index (IVF/HNSW) and run the reranker on a GPU. **The cost is the language model, not the database.**

### Design takeaway: always run every search method together; don't make the model choose
Since search is cheap and the model is expensive, every search should run **all methods together**: keyword (BM25), exact phrase, and vector, fused with RRF, then reranked. Making the model pick one method at a time (as racers C2 and D do) costs accuracy and saves almost nothing.

**Evidence from this project:**
- **The small agent picked badly.** C2's most used tool was BM25 (176 calls), the weakest first tool: 40% eventual accuracy, against 49% when starting with hybrid.
- **Telling it to use hybrid helped the comparisons.** In C2 v2, the advice "hybrid by default" made hybrid the first tool in 191 of 200 questions, and comparisons rose from 60% to 95%.
- **The best racers never choose.** C1 calls `hybrid` on every step, and it scored highest.
- **The combined search costs little.** Search, reranker included, takes 1–2 s per question, against 24–96 s of model time. Running every method is effectively free.
- **The methods cover each other's blind spots:**
  - Exact phrase finds rare names, titles and numbers that vectors blur.
  - Vectors find paraphrases that keywords miss.
  - The reranker reads the merged candidates and keeps the truly relevant ones.

**Recommendation for a real app:** expose **one search call** that always runs keyword + phrase + vector, fused with RRF, then reranked. Let the model (or the code) decide only *what to search for*, never *how*. One database that serves all three kinds of query in one request, as LanceDB does here, makes this simple.

---

## 4. The racers

| Racer | Model | Who controls the process |
|---|---|---|
| **CB** closed book | Gemma | No search at all (memory check) |
| **A** basic RAG | Gemma | One vector search, one answer |
| **B** big agent | gpt-oss-120b (Groq) | The model runs its own tool loop |
| **C1** fixed steps | Gemma | Code runs the steps; the model fills small JSON forms |
| **C2** guardrailed agent | Gemma | The model picks tool + query each turn; code enforces rules |
| **D** free small agent | Gemma | The model runs its own loop, like B, with no guardrails |

Small models: **Gemma 4 E2B** (2B effective) and **Gemma 4 E4B** (4B effective, 8.0B parameters in total), both run with greedy decoding, so they are deterministic.

---

## 5. How the pipeline was improved: big model as designer, small model as worker

### The loop
The improvements came from repeating one loop, with a large AI model (Claude, used through Claude Code) doing the analysis and engineering:

1. **Run** the small model on all 200 questions (Kaggle GPUs).
2. **Diagnose.** Claude read the traces and grouped the failures:
   - which step went wrong
   - whether the needed pages were found
   - whether the answer was in what the model saw
3. **Design a general fix.** A rule or prompt change that would apply to any dataset. Fixes that were specific to this dataset were rejected (see below).
4. **Test** the fix on a few questions, then on all 200.
5. **Keep it or drop it** based on the numbers, including per question type.

The human (project owner) set the goals, made the decisions and kept the work honest. The main decisions were rejecting dataset-specific fixes and insisting on simple, general rules. The large model did the failure analysis, code changes and runs.

### What this is, and what it is not
- **No weights were changed.** There is no fine-tuning, no training and no LoRA anywhere in the results. The Gemma models are the public checkpoints.
- **The pipeline was adapted to the task.** Prompts, step structure and checks were improved by looking at the small model's mistakes on these questions. In research terms this is close to **LLM-driven prompt / pipeline optimisation**: a stronger model optimises a weaker model's pipeline instead of its weights. Related published work includes DSPy (a teacher model compiles prompts for a pipeline), OPRO ("Large Language Models as Optimizers"), APE ("LLMs are human-level prompt engineers") and ADAS ("Automated Design of Agentic Systems").
- **A useful name for it: "scaffold distillation."** The big model's understanding of the task is distilled into code and prompts rather than into weights. The big model is paid once, at design time, not on every question.

### The honest caveat
The fixes were designed while looking at failures on the **same 200 questions** they were scored on. There was no separate held-out test set. So the scores are somewhat **optimistic**: the pipeline has been tuned on this question set, even though no weights were.

What reduces (but does not remove) the concern:
- **Every rule is general.** None mentions specific questions, entities or answers.
- **We rejected a dataset-specific fix:** having code compare birth years for "who was born earlier" questions would almost certainly have raised the score, but it is tied to this dataset's question template, so we left it out.
- **The fixes transferred to a model they were not designed on.** They were designed from the 2B model's failures; the 4B model ran the same code unchanged and scored higher (Section 7).

**This was then tested.** On 60 fresh, locked questions the tuned versions kept their scores (2B 67%, 4B 75%), so the tuning was not fitted to the old questions (Section 8, "Fresh held-out test"). Section 10 discusses whether this approach counts as cheating.

---

## 6. Timeline of versions and what each taught us

### 6.1 Baselines (Gemma 4 E2B, 200 questions)

| Racer | Accuracy |
|---|---|
| Closed book | 2.5% |
| A, basic RAG | 32.5% |
| D, free small agent | 20.0% (13.9% before a format-tolerant JSON parser) |
| C2, guardrailed agent | 45.0% (35.6% when guardrails were only logged, not enforced) |
| **C1, fixed steps** | **62.5%** |

**Lesson 1:** for a 2B model, letting the model run its own loop is far worse than code control: 20% (D) and 45% (C2) against 62.5% (C1).

**Why C2 lost to C1:**
- It chose weaker tools. Its most used tool was BM25 (176 calls, the weakest first tool); hybrid got 97 calls.
- It tried to stop early: 129 answers rejected by the guardrails.
- It repeated searches: 43 blocked repeats.
- It ran out of turns: 20 forced answers.

**Why C1 lost to the big model:**
- It stopped at the middle entity, answering "Argentina" instead of "Buenos Aires".
- It was stuck on a wrong first hop (17% of questions).
- It misread passages that did contain the answer (12%).

A side finding: passages were shown cut at 90 words, but chunks are 140 words, so the last 50 words of every chunk were never shown. In 4 of the 160 chain questions the answer was hidden there.

### 6.2 C1 v2: question decomposition + grounded answers → 58.5%
- The model writes the steps once ("Where was Messi born?" → "What is the capital of **#1**?"). Code runs them in order, filling in #1, so the model cannot stop half-way.
- Every step returns a **quote + answer**. Code checks the answer actually appears in the passages; otherwise it searches once more with a wider query.
- A final answer equal to an intermediate answer is rejected once.
- Whole 140-word passages are shown.

**Result:**
- 3-hop questions improved: 42% → 52%.
- **Comparisons collapsed: 72% → 50%.** The final step returned a date instead of a name, or a name where "yes/no" was needed.

**Lesson 2:** a fix aimed at one question type can break another. Always look at results per type.

### 6.3 C1 v3: answer-shape check → 64.5%
- If the question is yes/no, the answer must be "yes" or "no".
- If it names options ("X or Y?", "Which of X and Y…"), the answer must be one of them.
- Otherwise the model gets one retry, with a "write the values you compare first" prompt.

**Checks before the run:**
- The rule never flags any of the 200 gold answers.
- It flagged 13 of v2's answers, all wrong.

**Result:**
- It fired on 13 questions and fixed 12.
- Every other answer was identical to v2's.
- Comparisons went from 50% to **80%**.

### 6.4 C1 v4: backtracking, anchored steps, facts-first → 67.0%
Failure analysis of v3 (71 wrong): about 40 never found a needed page, mostly because hop 1 picked the wrong entity (e.g. "Lush" instead of "Albert Heijn"); about 30 had every page but misread.

Four general changes:
1. **Backtrack:** if a step finds nothing, redo the previous step asking for a different answer.
2. **Anchor:** a step that names nothing from the question gets the question's names added to its search.
3. **Facts first:** the final form writes the facts it compares before answering.
4. **Stricter plan rules:** start from the most specific clue; no vague steps.

**Result:**
- 67.0% overall (2-hop 69%).
- **Backtracking never fired.** It triggers when a step answers "unknown", and the 2B model almost never says "unknown". It grabs *something*, even when wrong. The gain came from changes 2–4. It is a small gain: 24 questions fixed vs 19 broken, so part of it may be noise.

**Lesson 3:** a check that waits for a small model to admit a mistake will rarely fire. Small models don't know when they are wrong.

### 6.5 C2 v2: the same checks on the agent loop → 47.0%
C2 got whole passages, grounded answers, the shape check, tool advice ("hybrid by default, grep for exact names") and facts kept with their quotes.

**Result:**
- Comparisons went to 95%.
- Hybrid became its first tool in 191/200 questions.
- **Chain questions got worse** (2-hop 45% → 36%). It averaged only 1.5 searches per question, made 532 answer attempts and had 166 rejected. It doesn't know the basic chain routine: search, find the name, search the name.

**Lesson 4:** checks can stop bad answers, but they can't teach a small model how to run a search loop.

### 6.6 A bigger small model: Gemma 4 E4B (4B), same code
E4B has 8.0B parameters, which is too large in fp16 for a 16 GB T4, so it was loaded in **8-bit** (bitsandbytes).

| Code | 2B | 4B |
|---|---|---|
| C1 v3 (code runs the steps) | 64.5% | **72.0%** |
| C2 v2 (model runs the loop) | 47.0% | **65.5%** |

**Lesson 5:** a bigger model helps the loop much more than it helps code control. The loop gained +18.5 points against +7.5 for code control. The gap between the loop and code control shrank from 20 points to 6.5.

---

## 7. Why the 4B does better with the *same* code

Paired comparison, C1 v3, identical code, 200 questions (`scripts/analyze.py`, section 2):

| Behaviour | 2B | 4B |
|---|---|---|
| Plans that link steps with #1 | 30 | **160** |
| Steps answered "unknown" (honest "not found") | 2 | **88** |
| Found the first needed page | 175 | **193** |
| Found every needed page | 140 | **169** |
| Needed the answer-shape retry | 13 | **0** |
| Answered with the intermediate entity | 14 | **5** |

1. **It plans the chain correctly.** For "who played the character with the conical hairdo with white lightning streaks?":
   - The 2B planned backwards and answered "Bride of Frankenstein" (the character).
   - The 4B planned "Which character has this design?" → "Who played **#1**?" and answered **Elsa Lanchester**.
2. **It admits "unknown"**, which triggers the code's wider second search, so it finds more pages.
3. **It gives the right kind of answer** the first time.

**Is the difference real?**
- 4B fixed 31 questions the 2B got wrong.
- 4B broke 16 that the 2B got right.
- Two-sided sign test p ≈ 0.04: probably real, not overwhelming.

---

## 8. Results

### All 200 questions (`results/summary.md`)

| Setup | Model | Accuracy | 2-hop | 3-hop | Comparison | $ / 1K q* |
|---|---|---|---|---|---|---|
| Closed book | E2B | 2.5% | 2% | 0% | 8% | 0.12 |
| A, basic RAG | E2B | 32.5% | 31% | 30% | 40% | 0.24 |
| D, free agent | E2B | 20.0% | 19% | 2% | 45% | 2.16 |
| C2, guardrailed agent | E2B | 45.0% | 45% | 34% | 60% | 2.10 |
| C2 v2 | E2B | 47.0% | 36% | 32% | 95% | 3.12 |
| C1, fixed steps | E2B | 62.5% | 65% | 50% | 72% | 1.36 |
| C1 v3 | E2B | 64.5% | 65% | 52% | 80% | 2.49 |
| C1 v4 | E2B | 67.0% | 69% | 52% | 80% | 2.32 |
| C2 v2 | **E4B** | 65.5% | 67% | 36% | 98% | 10.84 |
| **C1 v3** | **E4B** | **72.0%** | **74%** | 46% | **100%** | 9.37 |
| B, big agent (31 q) | gpt-oss-120b | 77.4% | 75% | 67% | 100% | 0.73 |

\* Cost model (`scripts/report.py`):
- Gemma: LLM seconds × a T4 at $0.35/hour.
- gpt-oss-120b: Groq list price ($0.15 / $0.75 per million input / output tokens).

### Head-to-head on the 31 questions racer B answered (`results/analysis.txt`)

| Setup | Right |
|---|---|
| B, gpt-oss-120b, own loop | **24/31 (77%)** |
| C1 v3, **4B** | **23/31 (74%)** |
| C1 v4, 2B | 22/31 (71%) |
| C2 v2, 4B | 20/31 (65%) |
| A, basic RAG, 2B | 9/31 (29%) |
| Closed book, 2B | 3/31 (10%) |

### Using the big model less: routing
The small model's checks flag shaky answers: an ungrounded final answer, a rejected intermediate answer, a shape retry, an ungrounded step or a forced answer.

| | Answers with no flag | Flagged answers |
|---|---|---|
| 2B C1 v4 | 69% right (162 q) | 61% right (38 q) |
| 4B C1 v3 | **78% right** (139 q) | **59% right** (61 q) |

The 2B's flags barely separate good from bad answers; the 4B's do. Simulated on B's 31 questions:
- **The 4B alone: 23.**
- **The 4B, with flagged questions sent to the 120B: 24**, sending 10 of 31 (about a third) to the big model.
- **The 120B alone: 24.**

This is a simulation on a small sample, not a measured system.

### Fresh held-out test: the honest check (`results/fresh_test.md`)

The worry: the pipeline was improved by looking at mistakes on the same 200 questions it was scored on, so its scores could be inflated. To check:

1. **60 new questions** were made with the same recipe (`data_build/make_fresh.py`, seed 1). They use only articles that no old question, and no old candidate question, touched: zero overlap. The mix matches the original (33 two-step, 15 three-step, 12 comparison). Questions a model could answer from memory were removed.
2. **The question file was committed before any system ran on it** (commit `42ecd2b`, sha256 `5f0a388f…`). The scoring script was also committed before any results existed.
3. **Every setup ran once, with no changes.** Nothing was tuned on these questions.

We also checked that **no correct answer appears anywhere in the prompts or code**: all 200 old answers were searched for and none was found. The model receives only the question text (`rag/run.py`); the correct answer is attached afterwards, for marking only.

| Setup | Tuned on old questions? | Old questions | **Fresh questions** |
|---|---|---|---|
| No search, memory only (2B) | no | 2% | **2%** |
| Search once, then answer (2B) | no | 32% | **42%** |
| Step-by-step plan run by code, first version (2B) | no | 61% | **62%** |
| Step-by-step plan + answer-shape rule (2B) | yes | 64% | **67%** |
| Step-by-step plan + answer-shape rule + retry (2B) | yes | 67% | **67%** |
| Step-by-step plan run by code, first version (4B) | no | not run | **67%** |
| Step-by-step plan + answer-shape rule (4B) | yes | 72% | **75%** |
| Model searches on its own, with checks (4B) | yes | 66% | **72%** |
| Big 120B model searches on its own | no | 77% (31 q) | **63%** (52 of 60; Groq's daily limit stopped it) |

With 60 questions, each score is uncertain by about ±12 points.

**What it shows**
- **The tuned versions did not fall back on new questions.** If the rules had been fitted to the old 200 questions, they would have dropped toward the untuned version's level. They held instead: 2B 67% → 67%, 4B 72% → 75%. The rules are general.
- **The tuning gain is small and not statistically certain:** +3 points on the 2B (+7 fixed / −4 broken, p = 0.55) and +5 on the 4B (+10 / −5, p = 0.30). The **step-by-step structure itself** is the big gain: 42% for "search once" vs 62% for the untuned step-by-step version on the same 2B.
- **The small models matched or beat the 120B agent here.** On the 52 questions both answered: 4B tuned 40, 4B untuned 35, 2B tuned 37, 120B 33. None of these gaps is statistically certain (4B tuned vs 120B: p = 0.12).
- **Why the 120B dropped:** 16 of its 19 misses were "unknown". It gave up after about 2 searches (on the old 31 questions, only 4 were "unknown"). That is a weakness of the simple agent loop we gave it, which was never tuned. A better prompt or more search budget would probably lift it. So the accurate claim is "the small model with code matched our 120B agent", not "the small model beats the 120B model".
- **Three-step questions:** on this set the 120B got 7/14 and the 4B 9/15. The earlier finding that "the 120B is clearly better at three steps" did not repeat.
- **Grading is strict:** "Chișinău" for "Chișinău, Moldova" and "Monterrey" for "Monterrey, Nuevo León" count as wrong. This affects every setup alike.

---

## 9. Conclusions

### Strong (large margins, consistent)
1. **For small models, code control beats an agent loop.** At 2B: 67% vs 47%.
2. **A big model can improve a small model's pipeline without training.** C1 went from 61–62.5% to 67% at 2B, and every rule is general.
3. **The same code works better with a slightly smarter worker**: 2B → 4B gave 64.5% → 72% with unchanged code. The rules were designed on the 2B and still helped the 4B.
4. **Agent loops need model intelligence.** The loop vs code gap went from 20 points (2B) to 6.5 (4B).
5. **Search is cheap; the language model is the cost.** Search is about 2% of time per question.

### Promising (small sample)
6. **Small model + code control is in the same range as a 120B agent.** Old set: 23 vs 24 of 31. Fresh locked set: 4B 40, 2B 37, 120B 33 of 52. The 120B's simple agent loop often gave up, and it was never tuned.
7. **Routing can cut big-model use**: it matched the 120B while sending about a third of the questions to it (simulated, 31 questions).

### Not shown
- **Cheaper.** Run one question at a time on a T4 (and in slow 8-bit for the 4B), the small models cost *more* per question than the 120B on Groq, which batches many users. A batched server (e.g. vLLM) is needed to measure real serving cost.
- **Generality.** One kind of document (new Wikipedia articles) and three question shapes. The 120B's loop handles any question; C1's fixed steps only handle what the code anticipates.
- **3-hop questions**: on the old set the big model was clearly better (67% vs 46% for the 4B), but on the fresh set it wasn't (120B 7/14, 4B 9/15). Not settled.

### Why use a small model with code at all?
- **The big model is paid once, at design time**, not for every question.
- **Runs on your own hardware**: one ordinary GPU, so documents stay private.
- **No rate limits, no API changes.**
- **Auditable**: every answer comes with its steps, quotes and the checks it passed.
- **The big model stays available as a backup** for flagged questions.

**Best fit:** many similar questions over a fixed document collection (support manuals, internal knowledge bases, contracts). **Poor fit:** open-ended assistants.

---

## 10. Discussion: is using a big model to optimise a small model's pipeline cheating?

### What the approach is called
There is no single standard name. Common ones:
- **LLM-driven prompt / pipeline optimisation**, or **"LLM as optimiser"**: a stronger model improves a weaker model's prompts and workflow.
- **Compound AI system optimisation**: the system is the model plus code, search and checks, and the whole thing is optimised, not just the model.
- **"Compiling" a pipeline**: DSPy's term. A teacher model writes and selects the prompts and examples a student model's pipeline uses.
- **"Scaffold distillation"**: our own descriptive term, not an established one. The big model's know-how ends up in the code and prompts around the small model instead of in its weights.

Related work: DSPy, OPRO ("Large Language Models as Optimizers"), APE ("LLMs are human-level prompt engineers"), ADAS ("Automated Design of Agentic Systems").

### An analogy
A hospital can have its senior doctor see every patient: excellent, but expensive and slow. Or the senior doctor can study the common cases, write checklists, and let junior doctors follow them for routine patients, with unusual cases still going to the senior.

| Analogy | This project |
|---|---|
| Senior doctor | Big model |
| Junior doctors | Gemma 2B / 4B |
| Checklist | The code steps, checks and prompts |
| Unusual cases to the senior | Routing flagged answers to the 120B |

### How the rules came from the mistakes
Each rule was the big model turning a pattern in the small model's mistakes into a general check. The rules contain no answers, only how to avoid typical mistakes.

| Mistake seen in the 2B's answers | Rule written into the code |
|---|---|
| Stopped half-way ("Argentina" instead of "Buenos Aires") | Split the question into linked steps; code passes step 1's answer into step 2 |
| Made up answers not in the passage | The answer must appear in the passage, or search again |
| Answer hidden in the last 50 words of a passage that were never shown | Show the whole 140-word passage |
| Gave a date when asked "who was born earlier, X or Y?" | The answer must be yes/no or one of the named options |
| Searches lost the main name ("the novel" instead of its title) | Keep the question's names in every search |

### Is it cheating?
**No.** Studying a system's mistakes and adjusting prompts, rules and code is ordinary error analysis and prompt engineering, which engineers have always done. Having a big AI do the analysis instead of a person changes the tool, not the nature of the work.

It **becomes misleading** in four situations:

1. **Tuning and testing on the same questions, then claiming it works on new ones.** This applied to us at first: the 200-question scores are **development-set scores**. We then ran a locked set of 60 fresh questions, and the tuned versions held their scores (Section 8), so the claim now rests on a real test.
2. **Rules that leak answers**, e.g. "if the question mentions X, answer Y". We did not do this, and we rejected the one rule that leaned that way: code comparing birth years for "who was born earlier" questions.
3. **Hiding the big model's role or cost.** Designing the pipeline took real big-model effort. It is a one-off cost, but it is part of the cost.
4. **A lopsided comparison.** The small model's pipeline got many design rounds, while the 120B ran its default loop with no tuning. A fully fair comparison would give both the same design effort, and the 120B would probably improve too.

These notes state 1, 3 and 4 openly.

**Fine-tuning vs this approach.** Fine-tuning changes the model's weights; this approach changes only the code and prompts around the model. No weights were changed here. The accurate claim is not "the model learned the dataset" but "the pipeline around the model was adapted to the task".

### Assessment

**Strengths**
- The expensive model is used where it adds the most value: understanding the problem once, not answering every question.
- The result is cheap to run, private and inspectable. The know-how is plain code and prompts anyone can read and change.
- It works with any small model. A better small model plugged into the same pipeline improved immediately (2B → 4B: 64.5% → 72%).

**Weaknesses**
- **Narrow.** The pipeline is good at the question types it was designed around; new types need another design round.
- **Easy to overfit.** The designer sees the mistakes, so without a locked test set you cannot tell general fixes from memorised ones.
- **The hardest reasoning may still need a big model.** On 3-hop questions the 120B won on the old set (67% vs 46%) but not on the fresh set, so this is unclear.
- **Diminishing returns.** Each round of fixes gained less than the one before: +6 points (v2 → v3), then +2.5 (v3 → v4). Past a point, a better small model or training helps more than more rules.

**Verdict.** A sensible, practical approach, and for a fixed, high-volume task arguably the right one. Think of it as **engineering the system**, not making the model smarter. Its main risk is overfitting, not cheating, and the cure is simple: a locked test set and an honest description of what was tuned.

### One-sentence summary for the blog
*"I used a large AI as the engineer and a small AI as the worker. The large model studied the small model's mistakes and redesigned the pipeline around it, with no change to its weights, On 60 fresh questions nobody had looked at, the 4B scored 75% and the 2B 67%, against 63% for a 120B agent."*

---

## 11. Limitations and how to test fairly

| Limitation | Fix |
|---|---|
| Pipeline improved on the same 200 questions it is scored on | **Done:** 60 fresh locked questions; the tuned versions held their scores (Section 8) |
| Big model only on 31 questions | **Mostly done:** 52 of 60 fresh questions (Groq daily limit). Finish the last 8 |
| The 120B's agent loop was never tuned and gives up early | Give it the same design effort (better prompt, larger search budget) before claiming the small model beats it |
| 4B and 120B not checked closed-book | Run both with no search on the locked set; set aside questions they answer from memory |
| Automatic answer matching unchecked | Hand-check a random 30–50 answers |
| Unbatched serving makes small models look expensive | One vLLM batched run on a T4 to measure real cost per 1K questions |
| 60 questions can only detect gaps of about 15+ points | State the claim as "within X points", not "equal" |

**Future work:** distillation (fine-tune the 2B on the 120B's or the 4B's correct traces, on separate training questions) to see whether a small model can *learn* to run the loop.

---

## 12. Infrastructure notes and problems we hit

- **Compute:**
  - Modal needed a payment method, so all GPU work ran on **Kaggle** (free, 2× T4 16 GB, ~30 GPU-hours/week).
  - Each job is a folder in `kaggle_jobs/` that clones this branch and runs `scripts/kaggle_small_racers.sh`, sharding questions by id across the two GPUs.
  - Kaggle allows **2 GPU jobs at a time** and stops jobs at 12 hours.
- **GPU out-of-memory on Kaggle:**
  - **Root cause:** Kaggle's image ships TensorFlow and JAX. When a library imported them, XLA pre-allocated most of the GPU memory.
  - **Fix:** `USE_TF=0 USE_JAX=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and related settings. We also load Gemma before the search models, skip the loader warm-up, and use expandable CUDA segments.
- **Identical index everywhere:** chunk embeddings are precomputed once and shipped (`data/emb_bge_small_f16.npy`), so every racer searches the same index and the GPUs only embed queries.
- **Big model via Groq free tier:**
  - The limit is 200K tokens/day per model.
  - Racer B runs in a quota-aware loop (`scripts/run_b_until_done.sh`) that resumes after the limit resets.
  - It recovers from Groq tool-call JSON errors.
  - The loop stopped several times when the container went idle, which is the main reason B only reached 31/60.
- **JSON from small models:** D's strict parser lost many answers (13.9%); a format-tolerant parser raised it to 20%. All forms ask for small JSON objects and code validates them.
- **E4B on a T4:** 8.0B parameters do not fit in fp16, so it was loaded in 8-bit (`GEMMA_QUANT=8bit`). This is about 4× slower per question (96 s vs 24 s), which is why its cost per question is high.
- **Prompt templating bug:** prompts containing JSON braces broke `str.format`; the plan prompt uses `.replace("{q}", …)` instead.

---

## 13. Reproducing

Code is on branch `claude/gallant-hamilton-rgm4j1`.

```bash
# build data (Groq for question generation)
python data_build/fetch_wikipedia.py --out data/ --days 29 --n-new 700 --seed 0
python data_build/make_questions.py --corpus data/corpus.jsonl.gz --graph data/graph.json.gz --out data/

# small-model racers on Kaggle (see kaggle_jobs/small_racers*/run.py for the exact commands)
kaggle kernels push -p kaggle_jobs/small_racers

# big agent (Groq)
scripts/run_b_until_done.sh

# reports
python scripts/report.py     # results/summary.md, summary.json, accuracy_vs_cost.png
python scripts/analyze.py    # results/analysis.txt (head-to-head, 2B vs 4B, routing)
```

Raw outputs, with every step, query, passage id and model output, are in `results/raw/*.jsonl`.

| File | Contents |
|---|---|
| `gemma-e2b-*.jsonl` | 2B runs (original tools, LanceDB, v3, v4, C2 v2) |
| `gemma-e4b-lance.jsonl` | 4B runs (C1 v3, C2 v2) |
| `B-gpt-oss-120b.jsonl` | big-model agent |

### Racer code (`rag/racers.py`)

| Racer | Class |
|---|---|
| C1 | `RacerC1` |
| C1 v2 | `RacerC1v2` |
| C1 v3 | `RacerC1v3` |
| C1 v4 | `RacerC1v4` |
| C2 | `RacerC2` |
| C2 v2 | `RacerC2v2` |
