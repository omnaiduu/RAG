[Code on GitHub](https://github.com/omnaiduu/RAG/tree/claude/gallant-hamilton-rgm4j1)

For the last few weeks I've been working on one question: can a small language model answer questions from a pile of documents about as well as a very big one, if I give it the right help?

The small model is Gemma 4 E4B. It does roughly the work of a 4 billion parameter model per word. The big one is gpt-oss-120b, which has 117 billion parameters. Small models are cheap to run on your own GPU and keep your documents private. Big models are smart but expensive, and you usually reach them through someone else's API.

Let me give you the result first. The rest of the post is how I got there and what I learned on the way.

| On a final test of 347 new questions, not used while building | Result |
| --- | ---: |
| Direct, one-lookup questions ("How many elevators are in JR Gate Tower?") | 86% right |
| Multi-step questions (find one thing, then use it to find another) | 68% right |
| Overall | 73% right |
| Wait time for one user, easy / hard question | 1.8 s / 4.4 s |
| Cost per 1,000 answers on one rented A10 GPU | about $0.66 |

On a separate set of 60 new multi-step questions, which I hadn't used while building anything, the small model with my code matched the 120B model running its own search loop (I'll explain the honest version of that claim later). The model's weights were never changed. No training, no fine-tuning. Everything came from the code around it.

I'll go through it from scratch, so you can follow even if you've never built one of these.

## What I was trying to do

The idea is called RAG, retrieval-augmented generation. Big word, simple idea:

:::steps
1. Someone asks a question.
2. You search your own documents for passages that might hold the answer.
3. You give those passages to a language model along with the question.
4. The model reads them and answers.
:::

The model doesn't need to know the answer. It only needs to read well.

This works nicely for direct questions. It gets hard when the answer needs two or three lookups in a row. For example:

```text
In the description of the second season of Hit and Miss Tour, the show travels
to a place that became a British colony after the Qing dynasty ceded it.
In what year range?
```

You can't search for the answer directly. First you have to find out where the show went (Hong Kong). Only then can you search for when Hong Kong became a British colony (1841–1842).

The popular way to handle this is "agentic RAG". You give the model a search tool and let it run in a loop: search, read, decide what to search next, and stop when it thinks it's done. Big models are good at this. Small models, as I found out, are not.

So my hypothesis was simple:

:::callout{tone="idea"}
A small model can get close to a big agent if plain code does the parts small models are bad at (planning the steps, deciding when to search again, checking answers), and the model only does small reading jobs.
:::

## The test data

I needed documents the models couldn't have memorised. Otherwise you can't tell if the model is reading or just remembering.

So I took **711 English Wikipedia articles created between 9 October and 7 November 2026**, newer than the models' training data, plus the articles they link to. That's 4,102 documents. I cut them into passages of 140 words each (with 30 words of overlap so a sentence isn't lost at the cut), which gave **64,262 passages**.

Then I made questions in the style of HotpotQA, a well-known multi-hop dataset:

| Type | What it needs | Example shape |
| --- | --- | --- |
| Bridge (2 hops) | Article A mentions B. The question describes B only through A and asks a fact from B | "Who played the character whose design includes a conical hairdo, as mentioned in the Vasaria article?" |
| Bridge (3 hops) | A chain A → B → C | "Which instruction set does the processor core that Google Axion uses implement?" |
| Comparison | Two articles | "Who was born earlier, X or Y?" |

Two other models (gpt-oss-20b and Qwen3.8-27B) wrote the questions from short excerpts. Code then checked each one: the answer must be copied from the last article, the middle entity must not be named in the question, and so on. Finally, I asked those models to answer every question **without** any documents. If either got it right from memory, the question was thrown out.

As a sanity check, the small model with no search at all scored **2.5%**. So the answers really weren't in its head.

## How search works here

Before the model reads anything, search has to find the right passages. I use three kinds of search together.

**1. Keyword search (BM25).** This counts shared words, and rare words count more. If your question has "Gymnocalycium" in it, a passage with that word is very likely relevant. "The" tells you nothing.

**2. Meaning search (vectors).** Each passage is turned into a list of 384 numbers by a small embedding model (bge-small-en-v1.5). Passages that mean similar things get similar numbers, even with different words. So "the show travels to" can match "beginning with a trip to".

**3. Exact-name search.** The planner (more on that later) lists the exact names in the question, like `"Hit and Miss Tour"`, and we fetch passages containing that exact phrase. Vectors tend to blur rare names and numbers. This doesn't.

Each search gives a ranked list. To merge them, I use Reciprocal Rank Fusion (RRF). It's tiny:

```python
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))
```

A passage near the top of several lists ends up near the top of the merged list. No tuning of weights needed.

That gives about 30 candidates. Then comes the most important piece: the **reranker**. It's a small model (bge-reranker-base) that reads each candidate *together with the question* and scores how well it answers it. The search methods judge a passage on its own; the reranker reads it against your exact question, which is why it's slower and much better. We keep the top 5, and those 5 are all the language model ever sees.

All of this lives in one LanceDB table: the text, a keyword index, and the vectors. One query does keyword + meaning search with RRF inside the database.

### Does combining the search types actually help?

I measured it on the final test. "Found" means the needed article was among the 5 passages the model gets to read:

| Search | Easy questions (100) | Hard questions, first article (233) |
| --- | ---: | ---: |
| Keyword only | 100% | 94% |
| Meaning only | 100% | 97% |
| Exact name only | 85% | 85% |
| All three, merged | 100% | 98% |
| All three + reranker | 100% | 99% |

Combining helps a bit on hard questions, 94–97% up to 98–99%. On easy questions any search works.

But here's the thing that surprised me. Finding the right *article* was almost never the problem. Picking the right *passage* inside it was. When I removed the reranker and just used the merged order, accuracy fell from **150 to 117 out of 200**. Same articles found, worse passages shown.

| Reranker | Right (of 200) | Time for 200 questions |
| --- | ---: | ---: |
| bge-reranker-base (what I use, on GPU) | **150** | 282 s |
| bge-reranker-v2-m3 (bigger, newer) | 144 | 632 s |
| Qwen3-Reranker-0.6B (2025) | 146 | 1,254 s |
| None | 117 | 160 s |
| None, show the model 10 passages instead of 5 | 129 | 213 s |

So: a reranker is essential, bigger ones weren't better here, and showing the model more passages doesn't make up for it.

### Should the model choose which search to use?

No. In one setup I let the small model pick a search tool each turn. It picked keyword search most often (176 times), which was the weakest starting tool: 40% eventual accuracy against 49% when it started with the combined search. Search is cheap, so just run all of it every time. Let the model decide *what* to search for, never *how*.

### Is all this searching heavy on the database?

I was worried about this, so I timed each part of a single search on a 4-core CPU:

| Part | Time | Share |
| --- | ---: | ---: |
| Turn the question into a vector | 112 ms | 2% |
| Database: keyword + meaning search | 102 ms | 2% |
| Database: exact-name search | 5 ms | 0% |
| Reranker scoring 30 passages | 4,414 ms | 95% |

The database is cheap. 64,000 passages is small, nobody writes to it, and it runs inside the same machine. The reranker is the heavy part, so I moved it to the GPU, where it takes a fraction of a second. After that, across the whole pipeline, the language model is 85% or more of the time. **The cost is the model, not the database.**

## Same question, six ways to answer it

Once search is in place, the real design question is: **who decides what to search next, the model or my code?**

To find out, I ran the same 200 questions through six different setups. Same documents, same search, same questions. Only the way of working changed. Here's what each one does with the Hong Kong question from earlier.

**1. No search (memory only).** The model just answers from what it already knows. This is only a check that the answers aren't in its memory. It would guess a year.

**2. Search once, then answer.** Code searches with the whole question, shows the model the top 5 passages, and the model answers. Simple and cheap. Here the search would most likely find the Hit and Miss Tour article but nothing about Hong Kong's history, because the question never says "Hong Kong". So it can't answer.

**3. Big model runs its own loop (gpt-oss-120b).** The model gets a search tool and decides everything itself, in a loop: what to search, when to search again, when to stop. A good big model searches the show, reads "Hong Kong", searches Hong Kong's history, and answers.

**4. Small model runs its own loop.** Exactly the same as setup 3, but with the small Gemma doing the deciding. No help, no rules.

**5. Small model runs its own loop, with code checking it.** The model still decides what to search, but code blocks bad moves: it rejects an answer that isn't in any passage, blocks a repeated search, stops it from answering too early.

**6. Code runs the steps, the model only reads.** The model writes a plan once ("first find the place, then find when it became a colony"). After that, *code* runs each step in order: search, show the model 5 passages, take its answer, plug it into the next step. The model never decides when to search or when to stop. It only answers small reading questions in a fixed JSON format.

| | Who decides what to search next | Who decides when to stop |
| --- | --- | --- |
| 1. No search | nobody | — |
| 2. Search once | code (once) | code |
| 3. Big model, own loop | the model | the model |
| 4. Small model, own loop | the model | the model |
| 5. Small model, own loop + checks | the model, within code's rules | the model, within code's rules |
| 6. Code runs the steps | code, following the model's plan | code |

Results with the smallest Gemma (E2B, about 2B per word), on the 200 questions:

| Setup | Right |
| --- | ---: |
| 1. No search | 2.5% |
| 4. Small model, own loop | 20.0% |
| 2. Search once, then answer | 32.5% |
| 5. Small model, own loop + checks | 45.0% |
| **6. Code runs the steps** | **62.5%** |

Letting the small model run its own loop was the worst of the lot, worse even than searching once. Reading its traces, it stopped early, searched the same thing twice, and picked the weakest search tool. Adding checks helped (45%), but the checks can only block bad moves; they can't teach the model the basic routine of "find the name, then search for that name". Taking the decisions away from the model entirely was three times better than letting it run free.

Setup 3, the big model, comes into the picture later, in the section on testing it fairly.

:::callout{tone="result"}
For small models, code should run the process. Letting a 2B model run its own loop gave 20%. Code running the same model through fixed steps gave 62.5%.
:::

## The step-by-step pipeline, piece by piece

Here's what the winning setup does for the Hong Kong question.

**Step 1, the plan.** The model reads the question and writes the lookup steps as JSON. If a step needs an earlier answer, it writes `#1`:

```json
{
  "steps": [
    "What is the place the show travels to in the description of the second season of Hit and Miss Tour?",
    "In what year range did #1 become a British colony after the Qing dynasty ceded it?"
  ],
  "exact_terms": ["Hit and Miss Tour"]
}
```

**Step 2, each lookup.** Code searches for step 1's question, shows the model the top 5 passages, and asks for a small JSON reply:

```text
Answer the sub-question using only the passages. Reply with only JSON:
{"quote": "<copy the one sentence from the passages that contains the answer>",
 "answer": "<shortest answer, copied word for word from that sentence>"}
If the passages do not contain it, use "answer": "unknown".
```

The model replied:

```json
{
  "quote": "The second season ... premiered on January 6, 2024, beginning with a trip to Hong Kong.",
  "answer": "Hong Kong"
}
```

**Step 3, check.** This is plain code, no AI. Does the answer actually appear in the passages the model was shown? If not, it's probably made up, and we search again:

```python
def _grounded(self, answer: str, idxs: list[int], extra: str = "") -> bool:
    a = _norm(answer)
    if not a or GIVE_UP.search(answer):
        return False
    hay = [_norm(self.tools.texts[i]) for i in idxs] + [_norm(extra)]
    return any(a in h for h in hay)
```

**Step 4, fill in and continue.** Code replaces `#1` with "Hong Kong", so step 2 becomes "In what year range did Hong Kong become a British colony...". It searches again, now landing on the Hong Kong article, and the model replies `"answer": "1841–1842"`.

**Step 5, final answer.** The model gets the facts found so far and writes the final JSON, including a `quote` and a one-line `reasoning`.

That's 4 model calls and 2 searches. The model never decides the order of anything. It only reads a few passages at a time and copies out an answer. The code makes sure it can't stop half-way, can't skip a step, and can't make up a name that isn't in the text.

## Improving it: a big model as the engineer

I improved the pipeline in a loop, using Claude (through Claude Code) as the engineer and Gemma as the worker:

:::steps
1. Run the small model on all 200 questions.
2. Have Claude read the traces and group the failures: which step went wrong, was the right page found, was the answer in what the model saw.
3. Write a general fix. No rules about specific questions or answers.
4. Run again, check per question type, keep it or drop it.
:::

Some of the mistakes and the rules they turned into:

| Mistake | Rule in the code |
| --- | --- |
| Stopped half-way: "Argentina" instead of "Buenos Aires" | Split into linked steps; code passes step 1's answer into step 2 |
| Made up answers | The answer must appear in the passage, or search again |
| Answer was in the last 50 words of a passage, which we never showed | Show the whole 140-word passage |
| Said a date when asked "who was born earlier, X or Y?" | The answer must be yes/no or one of the named options |
| Searches lost the main name ("the novel" instead of its title) | Keep the question's names in every search |

That last-50-words one is a good example of why you read traces. I was cutting passages at 90 words for display, but the chunks were 140 words long. In 4 questions the answer sat in the part the model never saw.

One fix broke something else. Splitting questions into steps helped 3-hop questions (42% → 52%) but comparisons crashed from 72% to 50%, because the final step started returning a date instead of a name. The answer-shape check fixed 12 of those 13 cases. Since then I always look at results per question type.

The versions went 62.5% → 64.5% → 67% on the 2B.

### Small models don't know when they're wrong

I added a "backtrack" rule: if a step finds nothing, go back and ask the previous step for a different answer. On the 2B it **never fired**. The rule waited for the model to say "unknown", and the 2B almost never says "unknown". It just grabs *something*, even when it's wrong.

:::callout{tone="warn"}
Any check that waits for a small model to admit a mistake will rarely fire. Checks have to come from code: is the answer in the passage, is it the right shape, did we already use it as a middle step.
:::

## Same code, slightly smarter model

Then I swapped in Gemma 4 E4B (about 4B per word) with the exact same code:

| Setup | 2B | 4B |
| --- | ---: | ---: |
| 6. Code runs the steps | 64.5% | **72.0%** |
| 5. Small model runs its own loop + checks | 47.0% | 65.5% |

The rules were designed by looking at the 2B's mistakes, and they still helped the 4B. Looking at why:

| Behaviour, same code | 2B | 4B |
| --- | ---: | ---: |
| Plans that link steps with #1 | 30 | 160 |
| Steps that honestly said "unknown" | 2 | 88 |
| Found every needed page | 140 | 169 |

The 4B plans proper chains, and it admits "unknown", which triggers the wider search, which finds more pages.

Also notice the gap between setup 6 (code runs the steps) and setup 5 (the model runs its own loop). It's 17.5 points at 2B and 6.5 at 4B. Later, with a 26B model, it was zero. **Agent loops need model intelligence.** The smarter the model, the less the code matters.

## Was I fooling myself?

There's an obvious problem here. I improved the pipeline by looking at mistakes on 200 questions, and then scored it on the same 200 questions. That's like studying the exam paper and then taking the same exam.

So I made 60 new questions, using only articles that no earlier question had touched, and followed three rules:

:::steps
1. Save the questions in git **before** running anything on them. Git records the date and the exact content, so I can't quietly change a question after seeing how the model did.
2. Never use them while improving the pipeline. No reading their mistakes, no fixing rules for them.
3. Run every setup on them once, unchanged, and report whatever comes out.
:::

It's the same as sealing an exam paper before the students start studying. If the students then do well, it's because they learned the subject, not because they saw the paper. People call this a held-out test set.

| Setup | Practice set | Fresh set |
| --- | ---: | ---: |
| Search once, then answer (2B) | 32% | 42% |
| Step-by-step, first version (2B) | 61% | 62% |
| Step-by-step, tuned (2B) | 67% | 67% |
| Step-by-step, tuned (4B) | 72% | 75% |
| 120B model running its own loop | 77% (31 q) | 63% (52 q) |

If the tuning had just memorised the practice set, the tuned versions would have dropped on fresh questions. They held. Also, the biggest gain is the step-by-step structure itself (42% → 62% on the same 2B), not the tuning on top.

About the 120B: it only finished 52 of 60 because of a free API daily limit, and 16 of its 19 misses were "unknown". It gave up after about 2 searches. I never tuned its loop, while my pipeline got many rounds. So the fair claim is "the small model with code *matched* my 120B agent", not "beat the 120B model". With 60 questions, every score here has about ±12 points of wobble anyway.

I also searched every prompt and every line of code for the correct answers. None of them were there. The model only ever gets the question text.

## Serving it on one GPU

Next question: which model gives the most users per rupee on one GPU? This needed some background I didn't have.

### Every model has two sizes

| | What it is | What it decides |
| --- | --- | --- |
| Stored size | All the weights that must sit in GPU memory | Which GPU you need, and room left for users |
| Working size | The weights that actually compute for each word | Speed |

For a normal ("dense") model these are the same. For two newer designs they're not.

**Gemma's "E" models use per-layer embeddings.** E4B stores about 8B numbers but only about 4B do real work per word. The rest are lookup tables that each layer reads one small row from. On a phone those tables can stay in normal RAM. On a server GPU they all sit on the GPU, so E4B uses the memory of an 8B but does the work of a 4B.

**Mixture of experts (MoE).** Think of a hospital with 128 specialists where each patient sees only 4. The building is huge, but each visit is cheap. gpt-oss-120b stores 117B parameters but uses about 5B per word. So "4B vs 120B" was really closer to "4B vs 5B" in work per word. The 120B's real edge is how much it *knows*, and in RAG, search supplies most of the knowledge.

### Results on one A10 (24 GB), served with vLLM

vLLM is a server that batches many users' requests together on one GPU. I rented the GPU on Modal and sent all 60 fresh questions at once.

| Model | Method | Right (of 60) | Room for users |
| --- | --- | ---: | --- |
| E4B, 4-bit | Step-by-step | 41 | most |
| E4B, full precision | Step-by-step | 43 | about half |
| 26B-A4B (MoE), 4-bit | Step-by-step | 47 | barely fits |
| 26B-A4B (MoE), 4-bit | Runs its own loop + checks | 47 | barely fits |

Two things here. Compressing E4B to 4-bit cost about 2 questions and doubled the room for users. And the 26B MoE was the most accurate, at the same speed as E4B (same ~4B working size), but it hardly fits on a 24 GB card. For real traffic it needs a bigger GPU.

Earlier, running one question at a time on Kaggle's free T4, E4B cost about $9 per 1,000 answers. Batched on vLLM it was about $0.55–0.80 in GPU time, close to what the 120B costs through an API. **Batching changes everything about cost.**

## Things that didn't work

Probably the most useful part of this post.

### Thinking mode

Gemma 4 has a "thinking" mode where it reasons before answering.

| | Right (of 200) | Time | Words generated |
| --- | ---: | ---: | ---: |
| Thinking off | **150** | 282 s | 73K |
| Thinking on | 128 | 858 s | 843K |

Worse and 3× slower. "Unknown" answers went from 12 to 42: reasoning at length, the small model talked itself out of answers that were right there in the passages. For short reading jobs, the code is already doing the thinking.

### More rules

After reading more failure traces, I wrote four more fixes into the step-by-step code. One of them was "if the answer looks like a description, ask for the name instead". My test for "looks like a description" was too loose: 5+ words, or starts with "a" or "the". It fired on **58 of 200 questions**, 42 of which were already right, and caused 7 of the 8 new mistakes. "Kohei Fujimura" became "Fujimura".

With the four fixes it scored 148, against 150 without them. I made that one check stricter (only answers starting with "a" or "an") and got 149. No real gain either way, so I dropped the fixes.

:::callout{tone="warn"}
A check that fires on 30% of questions to fix a problem seen in 2% does more harm than good. Log every time a rule fires, and judge it by what it breaks as well as what it fixes.
:::

### Examples in the prompt

Next I tried few-shot prompting: adding worked examples to the prompts, made up by me, one for each mistake type. For example, "three links in the question need three steps".

It scored **154 against 160** (with the fixed grader, more on that below). The examples didn't teach the habit. Three-hop questions that got a proper 3-step plan went from 13 to 14 out of 50, basically unchanged. But two-hop questions that wrongly got 3 steps went from 3 to 11. The model copied the *look* of the examples (more steps) instead of the rule behind them.

That was also a cheap warning about fine-tuning. A $0.20 test told me a $5–8 training run might not pay off.

## Where the chain actually breaks

I traced every failed chain back to the first step that went wrong:

| First thing that went wrong | Count |
| --- | ---: |
| Step 1 | 28 |
| Step 2 | 7 |
| All middle steps right, final answer wrong | 13 |

Step 1 is the weak link, because step 1 has to decode the question's roundabout description. Later steps get a clean name to search for. Some real step-1 slips:

- **A list instead of one thing.** "...led to the rise of a band" → "Hongo, Ekkaia, Ictus, Blunt and URSUS". Step 2 can't search for five bands.
- **The wrong item of the right kind.** "The boxer Snipes defeated" → "Larry Holmes", who actually beat Snipes.
- **Not knowing what a description refers to.** "The database referenced in the song's coverage" → the model answered with an album catalogue number. The passage said "AllMusic", but never called it a database.

Plans were also often too short. 37 of 50 three-hop questions got a 2-step plan, and those scored 59%. The 13 with a proper 3-step plan scored 77%.

And errors compound. If each step is about 90% right, two steps give 81% and three give 73%. The measured numbers line up: comparisons (no chain) 95%, two-hop 73%, three-hop 64%.

## Checking the checker

At one point I looked at the answers my grader marked wrong:

| Model said | Expected | Grader |
| --- | --- | :---: |
| 16 | 16 teams | ✗ |
| 5.1 | 5.1 magnitude | ✗ |
| Monterrey | Monterrey, Nuevo León | ✗ |
| Cua | river Cua | ✗ |

All correct, all marked wrong. The grader accepted extra words in the model's answer, but not fewer.

I fixed it with one general rule for every question: accept a number without its unit word, a place without its region, a name without a generic word like "river". Never dates ("7 April 2024" stays whole) and never lists. I applied it to all 260 expected answers *before* looking at any model's answers, and the check itself caught three bugs in my first version:

```python
# what the first version of the rule would have accepted, caught by auditing every expected answer
"40,290"            -> "40"           # split a number at its comma
"January 30, 1850"  -> "January 30"   # dropped the year
"Ercole d'Este, Lord of San Martino"  # "Mar" in "Martino" was read as March
```

Every setup gained 6 to 11 questions out of 200, and no setup changed rank. The best version went from 75% to 80%. The lesson stayed with me: when a score looks off, check the grader before you check the model.

## A realistic test, and a router

All my questions so far needed 2–3 lookups. Real users mostly ask direct things like "what's the refund policy?". So I built a bigger test of **347 new questions**, including **100 easy one-lookup questions**, on articles nothing else had used. Same three rules as before: saved in git before any run, never used for improving the pipeline, run once.

Then I compared two paths:

- **Quick path:** search once with the user's own question, read, answer. One or two model calls.
- **Full chain:** the step-by-step pipeline.

| Question type | Quick path | Full chain |
| --- | ---: | ---: |
| Easy, one lookup (100) | 86% | 82% |
| Two-hop (200) | 46% | 68% |
| All 347 | 60% | 73% |

Easy questions don't need the chain. Hard ones do.

I tried "quick path first, full chain only if it's unsure". It didn't work. The quick path was confidently wrong on hard questions and flagged only 42 of its 124 misses.

What did work: let the **planner** decide. It already writes the steps, so if the plan has one step, take the quick path:

```python
# The router, slightly simplified from the repo (class RacerR in rag/racers.py)
class Router(StepByStep):
    def solve(self, question):
        run = Run()
        plan = self.ask_json(run, PLAN_PROMPT.replace("{q}", question))
        steps = [s for s in plan.get("steps", []) if s.strip()]
        if len(steps) <= 1:
            # one lookup: search with the user's own question, read, answer
            answer = self.lookup(run, question)
            return self.finish(run, question, answer)
        # more than one lookup: the full step-by-step chain, reusing the same plan
        return super().solve(question, run=run, plan=plan)
```

It sent 85 of the 100 easy questions down the quick path and only 4 of the 247 hard ones.

| | Full chain only | With router |
| --- | ---: | ---: |
| Overall | 253 of 347 (73%) | 253 of 347 (73%) |
| Prompt tokens per easy question | 2,330 | 1,583 |

At first the router looked more accurate on easy questions, 86% vs 82%. Then I checked the 4 questions it "fixed". Three of them, the full chain had also answered right, just as "nine" when the expected answer was "nine stages". Spelled-out numbers, a gap my grader still has. So the honest result is: **same accuracy, about a third cheaper on easy questions.** On real traffic, where most questions are easy, that saving grows.

## How many users can one GPU serve?

Earlier I'd said around 175 requests fit on the GPU at once. That's true for *memory*, but it's the wrong measure. I measured actual waiting times with the router on one A10:

| Questions being answered at the same time | Easy question, typical wait | Hard question, typical wait | Answers per minute |
| ---: | ---: | ---: | ---: |
| 1 | 1.8 s | 4.4 s | 15 |
| 10 | 5.3 s | 14.7 s | 44 |
| 60 | about 67 s (all) | | 53 |

From 10 to 60 at once, answers per minute barely moved (44 → 53). Only the waiting grew. The GPU is already busy at about 10–15 questions at once. It's like a shop with room for 175 customers but one cashier.

Where the time goes at 60 at once: about 10 s of search per question, 52 s of model time. The model spends most of it reading prompts, mostly the 5 retrieved passages per call.

So one A10 gives about **50 answers a minute, at about $0.66 per 1,000 answers**. Users who send a question every few minutes rather than nonstop share that capacity, so it's a lot more people than 50.

## What I'd tell someone building this

**Search**
- Run keyword, meaning and exact-name search together on every query, merge with RRF, then rerank. Don't let the model pick.
- The reranker matters most (+16 points). A bigger reranker wasn't better.
- The database isn't the bottleneck. The model is.

**Small models**
- Let code run the process. The model should only do small reading jobs with a fixed JSON reply.
- Make the model quote its source, and check the quote in code.
- Don't wait for a small model to say it's unsure. It won't.
- Thinking mode and more prompt rules can make things worse. Count what each rule breaks.

**Serving**
- Know both sizes of a model: stored (memory) and working (speed).
- 4-bit compression was almost free. Batching (vLLM) changed the cost by more than 10×.
- Measure throughput and waiting time under load, not just how many requests fit in memory.
- Send easy questions down a cheap path. The planner can be the router.

**Testing**
- Keep a set of test questions you never use while building. Save it before running anything on it, and run it only to check, never to tune.
- 60 questions swing by ±12 points. Count fixes against breaks, and use bigger sets.
- Check the grader.
- Test on questions that look like real use.

## Limits

- **One kind of document** (Wikipedia) and machine-written questions. Real company documents and real users would look different. The exact scores won't transfer; the lessons probably will.
- **The 120B agent was never tuned**, while my pipeline got many rounds. With the same effort it would likely improve.
- **The test questions were written by AI**, and some are oddly worded or ambiguous.
- **On the final test, some articles start up to 3 questions**, because I was running out of unused articles. Those questions are a bit related.
- **Multi-step questions are still the weak spot** at 68%. The 26B model does about 10–14 points better there, if you can afford the bigger GPU.

## What's next

Test it on real documents with real questions. That's the test that actually matters. After that, try a faster GPU for prompt reading, shorter passages, and maybe fine-tuning, now that I have a test big enough to measure it.

The whole thing cost about $14 of Modal credit and a lot of free Kaggle and Groq hours.

:::refs
Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", EMNLP 2018
Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", 2022
Press et al., "Measuring and Narrowing the Compositionality Gap in Language Models", 2022
Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009
Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009
Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding" (BGE models), 2023
Nogueira and Cho, "Passage Re-ranking with BERT", 2019
Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM), SOSP 2023
Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines", 2023
Yang et al., "Large Language Models as Optimizers" (OPRO), 2023
Zhou et al., "Large Language Models Are Human-Level Prompt Engineers" (APE), 2022
Hu, Lu and Clune, "Automated Design of Agentic Systems", 2024
:::
