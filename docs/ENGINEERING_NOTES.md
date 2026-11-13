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

---

## 14. Serving on one GPU: which model gives the most users per dollar?

### The goal (owner's words, paraphrased)
Serve as many users as possible, at the lowest cost, with the best accuracy, on one GPU. The step-by-step code is what lets a small model do the job (Section 8, fresh test), so every test here uses it. The open question is **which model to put inside it**.

### Background we had to get straight: two sizes per model
Every model has two different "sizes", and they control different things:

| | What it is | What it controls |
|---|---|---|
| **Stored size** (total parameters) | everything that must sit in GPU memory | which GPU you need, and how much room is left for users |
| **Working size** (active parameters) | what actually computes for each word | speed, i.e. throughput and cost |

Stored size also roughly tracks how much the model *knows*.

| Model | Stored | Working | Why |
|---|---|---|---|
| Normal ("dense") 8B | 8B | 8B | every parameter works on every word |
| Gemma 4 E4B | ~8B | ~4B | "E" = effective; most of the extra is per-layer lookup tables |
| gpt-oss-120b | 117B | ~5B | mixture of experts: each word goes to a few of many "experts" |
| Gemma 4 26B-A4B | 26B | ~4B | mixture of experts, the Gemma version |

**Per-layer embeddings (PLE), the trick inside Gemma's E models.** A normal model has one embedding table at the input: one row of numbers per token, about 262,000 rows. Gemma's E models add a small extra table for **every layer**. Stored, that is billions of numbers. Per word, only one small row per layer is fetched, with almost no maths. On a phone the tables stay in normal RAM and only the ~4B "brain" sits on the chip (about 3 GB): a small desk, with the big encyclopedias kept on the shelf. On a server GPU, the standard loader puts everything on the GPU, so **E4B uses about the memory of an 8B but does about the work of a 4B**.

**Mixture of experts (MoE).** Like a hospital with 128 specialists where each patient sees only 4. The building is huge (all experts must be stored), but each visit is cheap. That is why gpt-oss-120b is cheap per token on an API: it works like a ~5B per word but needs an 80 GB GPU to hold it. So "4B vs 120B" was closer to "4B vs 5B" in work per word. The 120B's edge is stored knowledge, and in RAG the search supplies most of the knowledge.

**A lesson about our own earlier numbers.** On Kaggle's 16 GB T4 the E4B only fitted in 8-bit (bitsandbytes), which unpacks weights on every step and is slow. So the earlier E4B cost estimate (~$9 per 1,000 questions, unbatched, 8-bit) was pessimistic for two reasons: no batching, and a slow 8-bit path.

**How throughput works on one GPU.** Throughput ≈ speed per user × users at once.
- **Speed per user:** set by the working size.
- **Users at once:** set by the memory left after the weights, which holds each conversation's scratch memory (the KV cache).
- **Rule of thumb:** memory decides what fits, working size decides speed, stored size (plus training) roughly decides smarts. That is why MoE models are often the sweet spot for one GPU.

### What we decided to test, and what we skipped
To save credits, only the necessary runs:

| Run | Why |
|---|---|
| **Gemma 4 E4B, 4-bit (Google's QAT w4a16)** + step-by-step code | current best (75% on fresh); need its real batched speed and cost |
| **Gemma 4 26B-A4B, 4-bit (QAT AWQ INT4)** + step-by-step code | same ~4B working size, ~3× the stored knowledge: could raise accuracy at about the same speed |

**Skipped, with reasons:**
- **Gemma 4 12B (dense):** about 3× slower per word than both; the 26B-A4B is the better bet for "more accuracy, same GPU".
- **"Model searches on its own" runs:** already lost to the code approach.
- **31B:** too slow and costly for the goal.
- **Prompt tuning:** only after we know which model wins, and then tuned on the old 200 questions and checked on fresh ones.

**Setup.** Both models run on one Modal L4 (24 GB), served by vLLM, with all 60 fresh questions sent at once (60 simultaneous "users"). Search runs in the same container. The fresh questions were frozen before any of this.

**Recorded per model:** accuracy, answers per minute, p50 and max latency, tokens per second, peak GPU memory, vLLM's estimate of how many requests fit at once, and cost per 1,000 answers.

### Making Modal work from the cloud sandbox
- **Problem:** Modal's client talks gRPC (HTTP/2) to `api.modal.com`, and the sandbox's HTTPS proxy refused it ("Could not connect to the Modal server").
- **Fix:** a web search found that the official Modal client gained HTTP-proxy support in mid-2026, as an optional extra. `pip install "modal[api-proxy-support]"` makes it tunnel through the standard `HTTPS_PROXY`.
- **One more host:** jobs then also needed `input-plane.us-east.modal.com`, which was added to the environment's allowed hosts.
- **First job failure:** vLLM's FlashInfer sampler tried to compile CUDA code at start-up (`Could not find nvcc`) in a slim image. Fixed with `VLLM_USE_FLASHINFER_SAMPLER=0`; the sampler is not needed for greedy decoding.

### Results (one A10 24 GB GPU, all 60 fresh questions sent at once, vLLM)

| Model | Method | Accuracy | Full-length users that fit at once* | Time for 60 | $ per 1K answers (GPU + 16 CPU) | GPU only |
|---|---|---|---|---|---|---|
| E4B 4-bit | step-by-step code | 41/60 (68%) | **~27** | 134 s | $1.31 | $0.68 |
| E4B 4-bit | loop (searches on its own, with checks) | 38/60 (63%) | **~27** | 112 s | $1.09 | $0.57 |
| E4B full precision | step-by-step code | 43/60 (72%) | ~13 | 158 s | $1.54 | $0.80 |
| **26B-A4B 4-bit** | step-by-step code | **47/60 (78%)** | ~1 | 146 s | $1.42 | $0.74 |
| **26B-A4B 4-bit** | **loop** | **47/60 (78%)** | ~1 | 109 s | $1.07 | $0.56 |
| *(earlier)* E4B 8-bit, HF, Kaggle T4, one at a time | step-by-step code | 45/60 (75%) | — | — | ~$9 | — |
| *(earlier)* gpt-oss-120b via Groq API | its own loop | 33/52 (63%) | — | — | ~$0.73 | — |

\* vLLM's own estimate: KV-cache room ÷ 6,144 tokens per request. Our requests are much shorter (~1–2K tokens), so more fit in practice, but the ratio between models holds.

Prices are Modal list prices (A10 $0.000306/s, CPU $0.0000131/core/s, memory $0.00000222/GiB/s). The 16 CPU cores run search and are about half the bill; a deployment would size them to need. Cost per 1K is measured at a light load (60 at once); at heavier load the model with the most free memory pulls ahead.

**What it shows**
1. **The smarter MoE model is the most accurate: 26B-A4B gets 78%, with code or loop.** It does about the same work per word as E4B, so it runs at the same speed.
2. **But on a 24 GB GPU it has almost no room for users.** Its 4-bit weights are ~17 GB and it needs more memory per user (more layers), so vLLM estimates about 1 full-length conversation at once vs ~27 for E4B 4-bit. It only kept up here because our requests are short. For real multi-user load it needs a 48 GB+ GPU.
3. **With a smart enough model, the loop catches up with the code.** At 2B the loop was 20 points behind; at 4B, 3–5 points; at 26B-A4B, equal (47 vs 47), and the loop used fewer tokens and finished faster. The loop is more general, since it is not limited to question shapes the code anticipated.
4. **4-bit compression cost E4B little:** 41 vs 43 out of 60, within noise, for about twice the room for users (27 vs 13).
5. **Serving through vLLM changes the numbers.** E4B's 4-bit vLLM score (68%) is below its earlier 8-bit HF score (75%) on the same questions. Part is compression and part is numerical differences in a different engine; with 60 questions, ±12 points is normal wobble.
6. **Cost:** every setup lands at about $0.55–0.80 per 1,000 answers in GPU time at this light load, about the same as the 120B through Groq. The real saving appears only under heavy load, which is the next (and last) measurement.

**Spend so far on Modal:** about $2.

**Recommended next step (one run):** take the two leaders, E4B 4-bit (most users) and 26B-A4B (most accurate), and load each with about 4× more simultaneous requests: E4B on the A10, 26B-A4B on a 48 GB L40S so it has room. That measures real cost per 1,000 answers at full load.

### Prompt fixes for E4B 4-bit (tuned on the old 200, checked once on the fresh 60)

**Where E4B 4-bit went wrong** (19 misses on the fresh 60, step-by-step code):
- **Found the right article, answered "unknown": 5.** The answer was usually in a neighbouring passage it never saw.
- **Found the right article, picked the wrong fact: 6.** For example "1990" instead of "1987".
- **Never found a needed article: 7.** Usually an earlier step's answer was wrong, so the next search went astray.
- **Grader too strict: 1.**

**Fixes (general rules, nothing question-specific):**
- **v4, retry a stuck step:** if a later step finds nothing, ask the earlier step again for a *different* answer, then retry. Also write the facts before the answer. Built earlier for the 2B, where it never fired (the 2B rarely says "unknown"). E4B says "unknown" often, so it now fires.
- **v5, read wider (new):** if a step's answer is missing or not in the passages, the retry shows the passages just before and after the best hits in the same article, plus a wider search.

| Version (E4B 4-bit, A10, vLLM) | Old 200 (tuning set) | Fresh 60 (checked once) | Time for 60 fresh |
|---|---|---|---|
| v3 (before) | 140 (70%) | 41 (68%) | 134 s |
| v4 (+ retry) | 146 (73%) | not run | — |
| **v5 (+ retry + read wider)** | **150 (75%)** | **43 (72%)** | 166 s |

**Result.** Both question sets moved the same way: +5 points on the old 200 (17 fixed, 7 broken, p = 0.06) and +3 on the fresh 60 (4 fixed, 2 broken; too few to be statistically certain). The fix generalises, but the gain is modest. It costs about 25% more time per question, because the retries make extra searches and calls. The 26B-A4B still leads on fresh questions (47/60), so the remaining gap is mostly model ability, not prompts.

### The reranker: how much it matters, and whether a better one helps

**What it does.** Each search returns about 30 candidate passages, and a cross-encoder reranker (`BAAI/bge-reranker-base`, ~280M parameters) picks the 5 the model sees.

**Search was the bottleneck.** With 60 questions at once, each question spent ~83 s on search but only ~10 s in the language model.

| Part of one search (1 CPU core) | Time |
|---|---|
| Query embedding | 0.7 s |
| **LanceDB hybrid query** | **0.5 s** |
| Exact-phrase lookups | 0.02 s |
| **Reranking 30 passages** | **20.6 s** |

So the database was never the problem; the reranker on the CPU was. Moving the reranker to the GPU (one search at a time, each a few milliseconds) kept accuracy at 150/200 and cut the time for 200 questions from 494 s to 282 s.

**Audit.** Replaying all 162 searches E4B made on the fresh set: a needed *article* was in the reranker's top 5 in 161 of 162 cases. The reranker's value is choosing the right *passage* within articles.

| E4B 4-bit, v5 code, old 200 questions | Correct | Time for 200 |
|---|---|---|
| **bge-reranker-base (current, GPU)** | **150** | **282 s** |
| bge-reranker-v2-m3 (larger, newer) | 144 | 632 s |
| Qwen3-Reranker-0.6B (2025) | 146 | 1,254 s |
| No reranker (database's fused order, top 5) | 117 | 160 s |
| No reranker, top 10, the LLM picks | 129 | 213 s |

**Takeaways:**
- **A reranker is essential:** +33 questions, about 16 points.
- **Showing the LLM more passages instead only partly makes up for it,** and costs ~30% more input.
- **Bigger or newer rerankers were not better here,** and were 2–4× slower. Keep bge-reranker-base, run on the GPU.

### Thinking mode (Gemma 4 `enable_thinking`)

The model writes its reasoning before each JSON answer (up to ~1,000 extra tokens per call); code discards the reasoning and reads only the JSON.

| E4B 4-bit, v5 code, old 200 | Correct | Time for 200 | Tokens generated |
|---|---|---|---|
| Thinking off | **150** | **282 s** | 73K |
| Thinking on | 128 (36 broken, 14 fixed, p < 0.01) | 858 s | 843K |

**Thinking made it worse and 3× slower**, with 11.6× more tokens. Two reasons:
- **"Unknown" answers rose from 12 to 42.** Reasoning at length, the small model talked itself out of answers that were in the passages.
- **Some reasoning ran past the token budget**, so 12 questions got no answer at all.

For short, structured extraction steps, the plan-and-check code already does the "thinking" for the model; free-form reasoning on top adds cost and doubt rather than accuracy.

### Chain fixes (v6) from reading failure traces, and why one backfired

Reading the step-by-step traces of E4B v5's wrong answers (full examples in `docs/PIPELINE_EXPLAINED.md`, section 6) showed that most misses were **broken chains, not misreading**:

| Failure | Example | v6 fix (general rule) |
|---|---|---|
| Our "middle answer" rule rejected the right answer | Final "Leiden University" rejected because step 2 had found it; forced to "Adapt" | **Ask, don't forbid:** tell the model which step gave that answer; it may keep it if the question asks for exactly that |
| A vague step returned a description | "What is the description in 'Chocolate café' about?" → "a café that specializes in drinking chocolate" | **Name, not description:** re-ask for the specific name |
| Extra words broke the next search | "Ashoka Changemakers Fellow" → "Who founded Ashoka Changemakers Fellow?" → nothing | **Shorter form:** retry with "Ashoka Changemakers", then "Ashoka", before backtracking |
| A step forgot what it was about | "In what year was the original song mentioned … released?" (no #1) → found a different song | **Wider back-reference:** "mentioned / described / original …" adds the previous answer (not for comparisons) |
| The plan did not split the question | One step for a two-lookup question → "unknown" | not fixed |
| Ambiguous question or strict grading | "197" vs "approximately 190 countries" | not fixable fairly |

**Result on the practice set (E4B 4-bit, GPU reranker):**
- v5: **150/200**.
- v6: **148/200** (fixed 6, broke 8).

The trace flags showed why. "Name, not description" used a loose test ("5+ words or starts with an article") and fired on **58 of 200 questions**, 42 of them already correct. It was involved in **7 of the 8 breaks**: "Kohei Fujimura" became "Fujimura", and correct dates were swapped for other dates. The "shorter form" retry did what it should; it was part of 2 of the 6 fixes.

**v6b** keeps all four fixes but makes the name check strict: only answers starting with "a"/"an" (e.g. "a café that …") are re-asked. *(Result: see below when available.)*

**Lesson.** A check that fires on 30% of questions to fix a problem seen in 2% of them does more harm than good. Every rule should log when it fires, and be judged on both what it fixes and what it breaks.

### Could the model tell us *why* it chose something?

**Already logged:**
- **Every step reply has a `quote` field:** the exact sentence the answer came from.
- **The final reply has a `reasoning` field.**

These made the failure analysis above possible.

**Proposed: a debug-only `why` field** on plan and step replies ("why one step?", "why 1990 and not 1987?"), run once on the practice set. Cautions:
- **Small models often invent reasons after the fact,** so a "why" is a clue, not proof.
- **Asking for reasons can change the answers.** Full thinking mode cost 22 questions.
- **It adds tokens.**

**More reliable:** a large model reads the traces and classifies each failure, which is how the v6 fixes were found. It costs no GPU time.

### Is this test realistic? How often do such hard questions occur in real use?

**Fair as a comparison:**
- **Same questions, rules and grader for every setup.**
- **Answers can't come from memory:** the documents post-date the models, and memory-answerable questions were removed.
- **The exam set was locked before any run.**

**Not representative of typical real traffic:**
1. **Every question needs 2–3 lookups,** with the middle entity hidden on purpose ("the player he mentions"). Most real RAG questions (support docs, internal knowledge bases, reports) need **one** lookup. Multi-step questions are a minority: the hard tail.
2. **The wording is machine-generated and stilted.** A real user would ask "Where did Anthony Brooke die?", not "According to the account of the Malay National Union of Sarawak, where did the former Rajah Muda die?".
3. **Some questions are ambiguous or strictly graded.**
4. **The documents are Wikipedia,** not product manuals or company documents.

**What transfers to real use:**
- **Step-by-step code helps small models on hard questions.**
- **A reranker is essential.**
- **Run the reranker on the GPU, keep thinking off, and use 4-bit E4B** for users per GPU.
- **Rules must be judged by what they break as well as what they fix.**

**What does not transfer:** the exact scores. Real users would see higher accuracy on simpler questions.

**A risk this test hides.** For one-lookup questions the step-by-step pipeline may be overkill: about 5 model calls instead of 1 for "search once, then answer". It costs more and may over-complicate easy questions.

**Decision:** stop fine-tuning on this question set after v6b. Gains are now within noise (±2–3 questions), and further tuning would fit the test's quirks rather than real use.

### Next steps (as of this session)
1. **Easy-question set:** about 60 one-lookup questions, generated the same way (free with Groq). Check that step-by-step does not hurt easy questions, and measure how much cheaper "search once" is on them.
2. **A router for real use:** answer with "search once" first; switch to step-by-step only when the question needs several lookups or the first answer fails its checks (not grounded, "unknown", wrong shape). This is the realistic design for most users per dollar.
3. **Test on real data:** the owner's actual documents and 30–50 real questions, if available. Worth more than any further tuning here.
4. **Heavy-load cost test:** send about 4× more questions at once to E4B 4-bit on the A10, to measure real cost per 1,000 answers when the GPU is fully busy.

### Session state, for continuing later
- **Best setup:** E4B 4-bit (`google/gemma-4-E4B-it-qat-w4a16-ct`), C1v5, bge-reranker-base on the GPU, thinking off.
  - Practice set: 150/200.
  - Fresh 60: 43/60, with search on the CPU in that run.
- **Benchmark command (Modal, A10):**
  `GPU_TYPE=A10 modal run modal_jobs/bench.py --model google/gemma-4-E4B-it-qat-w4a16-ct --racer C1v5 --questions old --search-device cuda --mem-util 0.65 --tag <name>`
  - Options: `--questions fresh|old`, `--rerank <model|none>`, `--show-k N`, `--think`, `--racer C1v3|C1v4|C1v5|C1v6|C1v6b|C2v2`.
  - Outputs: `results/raw/vllm-<tag>-<set>.jsonl` and `results/bench/<tag>.json`.
- **Modal from this sandbox:** `pip install "modal[api-proxy-support]"` (official HTTP-proxy support). `input-plane.us-east.modal.com` must be in the environment's allowed hosts. The token lives in `~/.modal.toml`, never in the repo.
- **Spend:** about $9 of the $30 Modal credit used so far.
