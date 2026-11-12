# Fresh held-out test

  python scripts/fresh_report.py      # prints the table and writes results/fresh_test.md

"Old" = the 200 questions the pipeline was improved on. "Fresh" = data/fresh/questions.jsonl: 60 new questions on
articles no old question touched, frozen (committed) before any setup was run on them. Nothing was tuned on them.

| Setup | Tuned on old questions? | Old questions | Fresh questions (95% range) | Change |
|---|---|---|---|---|
| No search, memory only (2B) | no | 2% (200 q) | **2%** (1/60; 0%-9%) | -1 pts |
| Search once, then answer (2B) | no | 32% (200 q) | **42%** (25/60; 30%-54%) | +9 pts |
| Step-by-step plan run by code, first version (2B) | no | 61% (200 q) | **62%** (37/60; 49%-73%) | +1 pts |
| Step-by-step plan + answer-shape rule (2B) | yes | 64% (200 q) | **67%** (40/60; 54%-77%) | +2 pts |
| Step-by-step plan + answer-shape rule + retry (2B) | yes | 67% (200 q) | **67%** (40/60; 54%-77%) | -0 pts |
| Step-by-step plan run by code, first version (4B) | no | nan% (0 q) | **67%** (40/60; 54%-77%) | +nan pts |
| Step-by-step plan + answer-shape rule (4B) | yes | 72% (200 q) | **75%** (45/60; 63%-84%) | +3 pts |
| Model searches on its own, with checks (4B) | yes | 66% (200 q) | **72%** (43/60; 59%-81%) | +6 pts |
| Big 120B model searches on its own | no | 77% (31 q) | **63%** (33/52; 50%-75%) | -14 pts |
