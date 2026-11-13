# Router simulation on the exam

  python scripts/router_sim.py [quick_tag] [full_tag]     # defaults: q-e4b v5-e4b; writes results/router_sim.md

Rules compared (accuracy, and cost as model calls / prompt tokens per question):
  full only            every question takes the full step-by-step chain
  quick only           every question takes one search + one answer
  quick, then full     quick path first; if it flags itself unsure (unknown / not in the passages), run the full chain
                       too (cost = both)
  perfect router       easy questions -> quick, all others -> full (an upper bound: needs knowing the type)

| Rule | Easy (one lookup) | Hard (chains) | All | Model calls / question | Prompt tokens / question |
|---|---|---|---|---|---|
| Full chain only | 82/100 (82%) | 171/247 (69%) | **253/347 (73%)** | 4.3 | 4081 |
| Quick path only | 86/100 (86%) | 123/247 (50%) | **209/347 (60%)** | 1.2 | 1199 |
| Quick first, full chain if unsure | 86/100 (86%) | 148/247 (60%) | **234/347 (67%)** | 1.9 | 1901 |
| Perfect router (easy -> quick, hard -> full) | 86/100 (86%) | 171/247 (69%) | **257/347 (74%)** | 3.7 | 3676 |

Quick path flagged itself unsure on 42/347 questions (0 easy, 42 hard).

| Question type | Quick path | Full chain |
|---|---|---|
| bridge (200) | 92 (46%) | 136 (68%) |
| bridge3 (33) | 23 (70%) | 22 (67%) |
| comparison (14) | 8 (57%) | 13 (93%) |
| single (100) | 86 (86%) | 82 (82%) |
