# How the step-by-step pipeline works: problem, data, prompts, outputs, failures and fixes

This is the detailed companion to `ENGINEERING_NOTES.md`. Everything below is real: the prompts are copied from `rag/racers.py`, and the model outputs come from logged runs (`results/raw/`).

---

## 1. The problem

**Goal.** Answer questions that need **two or three lookups**, using only a fixed collection of documents, with a **small, cheap model**, serving many users at once on one GPU.

**Why it is hard for a small model.** It has to:
1. work out what to look up first;
2. search;
3. read the result and pull out the right fact;
4. use that fact to make the next search;
5. give the final answer.

Small models are decent at step 3, a short focused reading job, but bad at planning and keeping track (1, 4, 5). They stop half-way, lose the thread, or make things up.

**The approach.** Let **code** do the planning bookkeeping and the checking, and give the **model** only small, structured jobs.

---

## 2. The data

### Documents
- `data/corpus.jsonl.gz`: 4,102 Wikipedia articles, mostly created after the models' training, so the models must search.
- They are cut into **64,262 passages** ("chunks") of 140 words with 30 words of overlap.

One document:
```json
{"id": "d00000", "title": "2026 Henderson Metal Fall Classic",
 "text": "The 2026 Henderson Metal Fall Classic presented by Kioti Tractors of Ontario was held from October 21 to 24 at the YNCU Curling Centre in Sault Ste. Marie, Ontario. ..."}
```

How a passage looks when shown to the model (`[id] Title (Section): text`):
```
[c56980] Malay National Union of Sarawak: The Malay National Union of Sarawak (MNUS; Malay: Persatuan Kebangsaan Melayu Sarawak, PKMS) is a community organisation in Sarawak, Malaysia. ...
```

### Questions
**Old set, 200 questions (`data/questions.jsonl`):** the practice set we improved the pipeline on.

**Fresh set, 60 questions (`data/fresh/questions.jsonl`):** the exam. Built the same way from articles the old set never touched, and locked before any run.

```json
{"qid": "f000", "question": "According to the account of the Malay National Union of Sarawak, where did the former Rajah Muda die?",
 "answer": "Whanganui, New Zealand", "type": "bridge", "hops": 2,
 "support_titles": ["Malay National Union of Sarawak", "Anthony Brooke"]}

{"qid": "f001", "question": "In what year was the original song that 'All My Exes (Kacey's Version)' is based on released, as mentioned in the description of the album 'Tex Mex Tapes'?",
 "answer": "1987", "type": "bridge", "support_titles": ["Tex Mex Tapes", "All My Ex's Live in Texas"]}

{"qid": "f002", "question": "Who was born earlier, Francis Butterfield or Yevgeny Lukyanov?",
 "answer": "Francis Butterfield", "type": "comparison"}
```

Question types: **bridge** (2 lookups: A mentions B; the answer is in B), **bridge3** (3 lookups: A → B → C) and **comparison** ("who was born earlier, X or Y?").

---

## 3. The system, end to end

```
question
   │
   ▼
[1 PLAN]  model → JSON list of steps (#1 = "answer of step 1")
   │
   ▼  for each step, in order (code runs the loop):
[2 FILL]  code puts earlier answers into #1/#2 and adds names the step forgot
[3 SEARCH] code: keyword + vector + exact phrase (one LanceDB table) → ~30 candidates
           → reranker (bge-reranker-base) picks the best 5 passages
[4 READ]  model → JSON {quote, answer} from those 5 passages
[5 CHECK] code: is the answer really in the passages? if not → retry (read wider)
          does the next step find nothing? → retry with a shorter form, or redo the earlier step
   │
   ▼
[6 FINAL] model → JSON {quote, reasoning, answer, sources} from the facts found + passages
[7 CHECK] code: not just a middle step's answer (ask, don't forbid)? in the passages?
          right shape (yes/no, or one of the named options)?
   │
   ▼
answer
```

**Who does what:**
- **Code:** the loop, filling in #1, all searches, every check and every retry.
- **Model:** only steps 1, 4 and 6, each a short job with a fixed JSON reply.

There is **no system prompt**. Each model call is a single user message containing one of the prompts below. Decoding is greedy (temperature 0), so runs are repeatable.

---

## 4. The prompts (exact text) and what the model must reply ("structured output")

"Structured output" means the model must reply with a **JSON object in a fixed shape**, never free text. Code parses it. If it isn't valid JSON, code asks once: "That was not valid JSON. Reply with only the JSON object."

### 4.1 PLAN prompt
```
Break the question into the smallest lookup steps, in order (1 to 3 steps). Write each step as a short
question. If a step needs the answer of an earlier step, write #1 (or #2) in its place.
Rules: start from the most specific name or clue in the question. Every step must say what it is about:
copy the name from the question or use #1. Never write a vague step like "that character" or "the description".
Example: "What is the capital of the country where Messi was born?" -> {"steps": ["Where was Messi born?", "What is the capital of #1?"], "exact_terms": ["Messi"]}
Example: "According to the Inception article, who played the dream thief, and where was that actor born?" -> {"steps": ["Who played the dream thief in Inception?", "Where was #1 born?"], "exact_terms": ["Inception"]}
Example: "Which is taller, the Eiffel Tower or Big Ben?" -> {"steps": ["How tall is the Eiffel Tower?", "How tall is Big Ben?"], "exact_terms": ["Eiffel Tower", "Big Ben"]}
exact_terms: names or titles copied exactly from the question.
Reply with only JSON.

QUESTION: {question}
```
The model replies, for example:
```json
{"steps": ["Which specific organization named Olivier Nsengimana a fellow in 2023?", "Who founded #1?"],
 "exact_terms": ["Olivier Nsengimana"]}
```
The examples use Messi, Inception and Eiffel Tower / Big Ben, none of which appears in any test question.

### 4.2 STEP prompt (one per step)
```
Answer the sub-question using only the passages. Reply with only JSON:
{"quote": "<copy the one sentence from the passages that contains the answer>", "answer": "<shortest answer, copied word for word from that sentence>"}
If the passages do not contain it, use "answer": "unknown".{note}

SUB-QUESTION: {step with #1 filled in}

PASSAGES:
[c....] Title: 140-word passage
... (5 passages; up to ~10 when "reading wider")
```
The model replies, for example:
```json
{"quote": "He was named an Ashoka Changemakers Fellow in 2023.", "answer": "Ashoka Changemakers Fellow"}
```
`{note}` is empty normally. When a step is redone with "find a different answer", it reads: *The answer is not "X" (that led nowhere): give a different one that fits.*

### 4.3 FINAL prompt
```
Answer the question using the facts and passages. Reply with only JSON:
{"quote": "<the sentence or fact the answer comes from>", "reasoning": "<one short sentence: the facts you use; if the question compares, write both values>", "answer": "<shortest possible answer: a name, date, number or few words>", "sources": ["<passage ids>"]}
If the answer is not there, use "answer": "unknown".{note}

QUESTION: {question}

FACTS FOUND:
- Which specific organization named Olivier Nsengimana a fellow in 2023? -> Ashoka Changemakers Fellow (from: "He was named an Ashoka Changemakers Fellow in 2023.")
- Who founded Ashoka Changemakers Fellow? -> unknown

PASSAGES:
(last 10 passages seen)
```

### 4.4 SHAPE retry (only if the answer has the wrong form)
```
Answer the question from the facts. First write the values you compare, then the answer.
Reply with only JSON: {"reasoning": "<one short sentence>", "answer": "<the answer>"}
Note: This is a yes/no question: the answer must be "yes" or "no".   (or: choose one of the options it names)

QUESTION: ...
FACTS FOUND: ...
```

---

## 5. What the code checks (no model involved)

| Check | Rule | Version |
|---|---|---|
| **Grounded step** | the step's answer must appear in the passages shown, else retry | v2 |
| **Grounded final** | the final answer must appear in the facts, passages or question, else ask again once | v2 |
| **Answer shape** | a yes/no question must get yes/no; "X or Y?" must get X or Y | v3 |
| **Anchor** | a step that names nothing from the question gets the question's names added to its search | v4 |
| **Backtrack** | if a step built on an earlier answer finds nothing, redo the earlier step asking for a *different* answer | v4 |
| **Read wider** | when a step fails, the retry shows the passages just before and after the best hits in the same article | v5 |
| **Name, not description** | an answer later steps build on must be a name; if it looks like a description, ask again for the name | v6 |
| **Shorter form** | if the next step finds nothing, retry it with shorter forms of the earlier answer | v6 |
| **Wider back-reference** | "the original song mentioned…" without #1 gets the previous answer added (not for comparisons) | v6 |
| **Middle answer: ask, don't forbid** | if the final answer equals an earlier step's answer, tell the model which step and let it keep it if that's what the question asks | v6 (v2–v5 forbade it) |

---

## 6. Real runs (E4B 4-bit, v5), with the failure in each

### 6.1 The middle-answer rule rejected the right answer (fixed in v6)
**Q:** Rik Peels is a principal investigator in a research programme led by a historian who co-founded the Centre for Terrorism and Counterterrorism at which university? **Gold:** Leiden University
```
PLAN : ["Who is the historian who co-founded the Centre for Terrorism and Counterterrorism?",
        "What university is the Centre for Terrorism and Counterterrorism at?",
        "What is the research programme led by the historian ... at #2, in which Rik Peels is a principal investigator?"]
step1: answer "Beatrice de Graaf"   quote "In 2007 she transferred to Leiden University, where she co-founded the Centre ..."
step2: answer "Leiden University"   (same quote)
step3: answer "Adapt!"              quote "In March 2024 a consortium ... led by the historian Beatrice de Graaf ..."
FINAL attempt 1: "Leiden University"   ← correct
CODE: rejected: "Leiden University" was an earlier step's answer → "only an intermediate fact"
FINAL attempt 2: "Adapt"               ← wrong
```

### 6.2 A vague step returned a description (fixed in v6)
**Q:** According to the description in "Chocolate café", which city hosts a branch of the Madrid café mentioned there that is located in the United States? **Gold:** Austin, Texas
```
PLAN : ['What is the description in "Chocolate café" about?', 'Which city hosts a branch of the Madrid café mentioned in #1 that is located in the United States?']
step1: answer "a café that specializes in drinking chocolate"     ← a description, not a name
step2: query "Which city hosts a branch of the Madrid café mentioned in a café that specializes in drinking chocolate ..."
       answer "Madrid"   quote "Chocolatería San Ginés, ... in Madrid, opened in 1894 ..."
FINAL: "unknown"
```
Also: "What is the account of Rolando Camarero?" was answered with "Mexican professional baseball player and manager".

### 6.3 Extra words in an answer broke the next search (fixed in v6)
**Q:** Olivier Nsengimana was named a fellow of a specific organization in 2023; who founded that organization? **Gold:** Bill Drayton
```
step1: answer "Ashoka Changemakers Fellow"     ← the organisation is "Ashoka"
step2: query "Who founded Ashoka Changemakers Fellow?" → answer "unknown"
backtrack: step1 asked for a different answer → same answer
FINAL: "unknown"
```

### 6.4 A step forgot what it was about (fixed in v6)
**Q:** In what year was the original song that 'All My Exes (Kacey's Version)' is based on released, as mentioned in the description of the album 'Tex Mex Tapes'? **Gold:** 1987
```
step1: answer "George Strait's \"All My Ex's Live in Texas\""   ← right
step2: "In what year was the original song mentioned in the description of the album 'Tex Mex Tapes' released?"   ← no #1
       answer "1990"  quote "... '(Hey Baby) Que Paso' by the Texas Tornados, which appeared on their 1990 ... album"   ← a different song
FINAL: "1990"
```

### 6.5 The plan did not split the question (NOT fixed yet)
**Q:** According to the account of the Malay National Union of Sarawak, where did the former Rajah Muda die? **Gold:** Whanganui, New Zealand
```
PLAN : ["Where did the former Rajah Muda die according to the account of the Malay National Union of Sarawak?"]   ← one step only
step1: "unknown"  (the MNUS passages name Anthony Brooke but say nothing about where he died)
read wider: "unknown"
FINAL: "unknown"
```
The plan should have been "Who was the former Rajah Muda in the MNUS account?" → "Where did #1 die?". This happened once in 60. An earlier walk-through used this question as an idealised example; the real E4B v5 run got it wrong as shown here.

### 6.6 An ambiguous question (not fixable fairly)
**Q:** Based on the founder's legacy mentioned at the Columbus Engine Plant, how many countries does the company sell in? **Gold:** "approximately 190 countries". **Model:** "197" ("By 2013, Cummins had operations in 197 countries and territories.")

---

## 7. Current state

| E4B 4-bit, one A10 GPU | Old 200 (practice) | Fresh 60 (exam) |
|---|---|---|
| v3 | 140 | 41 |
| v4 (+ backtrack) | 146 | — |
| **v5 (+ read wider)** | **150** | **43 (72%)** |
| v6 (+ 4 chain fixes) | running | — |
| v5, no reranker | 117 | — |
| v5, thinking mode on | 128 | — |
| *26B-A4B, v3 (for reference)* | — | 47 (78%) |
| *gpt-oss-120b agent (for reference)* | — | 33/52 (63%) |

**Settings kept:**
- **Reranker:** bge-reranker-base, on the GPU. The larger ones were not better and were 2–4× slower.
- **Thinking:** off. Worse and 3× slower.
- **Model:** E4B 4-bit. About 2× the room for users vs full precision, and about 175 short requests fit at once on a 24 GB GPU.
