### Core 60 (every racer, including B)

| Racer | Model | n | Accuracy | F1 | All evidence found | LLM calls | Tool calls | Prompt tok | Output tok | LLM s/q | Tool s/q | $ / 1K q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: basic RAG | `gemma-4-E2B-it` | 60 | **33.3%** | 0.36 | 47% | 1.0 | 1.0 | 637 | 35 | 2.6 | 0.0 | 0.251 |
| B: big agent | `gpt-oss-120b` | 31 | **77.4%** | 0.78 | 65% | 3.77 | 2.42 | 3979 | 182 | 19.9 | 5.6 | 0.733 |
| C1: fixed steps | `gemma-4-E2B-it` | 60 | **50.0%** | 0.50 | 57% | 3.5 | 1.73 | 2598 | 194 | 14.4 | 1.0 | 1.401 |
| C1: fixed steps [lance] | `gemma-4-E2B-it` | 60 | **53.3%** | 0.54 | 62% | 3.53 | 1.78 | 2715 | 195 | 14.7 | 1.1 | 1.433 |
| C1 v2: steps + grounded answers [lance] | `gemma-4-E2B-it` | 60 | **55.0%** | 0.58 | 73% | 4.45 | 2.33 | 3991 | 313 | 22.9 | 1.5 | 2.226 |
| C2: guardrailed agent | `gemma-4-E2B-it` | 60 | **43.3%** | 0.46 | 58% | 3.57 | 1.68 | 2690 | 305 | 22.3 | 0.3 | 2.172 |
| C2 v1 (guardrails logged, not enforced) | `gemma-4-E2B-it` | 30 | **40.0%** | 0.43 | 43% | 2.27 | 1.1 | 1369 | 183 | 12.9 | 0.2 | 1.259 |
| closed book | `gemma-4-E2B-it` | 60 | **8.3%** | 0.12 | - | 1.0 | 0.0 | 72 | 18 | 1.3 | 0.0 | 0.125 |
| D: free small agent | `gemma-4-E2B-it` | 60 | **13.3%** | 0.20 | 57% | 2.83 | 1.83 | 2444 | 307 | 22.3 | 0.1 | 2.170 |
| D v1 (strict parser) | `gemma-4-E2B-it` | 30 | **13.3%** | 0.15 | 27% | 1.73 | 0.73 | 1065 | 163 | 11.5 | 0.0 | 1.121 |

Accuracy by question type (core60):

| Racer | Model | bridge | bridge3 | comparison |
|---|---|---|---|---|
| A: basic RAG | `gemma-4-E2B-it` | 33% (n=30) | 27% (n=15) | 40% (n=15) |
| B: big agent | `gpt-oss-120b` | 75% (n=16) | 67% (n=9) | 100% (n=6) |
| C1: fixed steps | `gemma-4-E2B-it` | 57% (n=30) | 27% (n=15) | 60% (n=15) |
| C1: fixed steps [lance] | `gemma-4-E2B-it` | 60% (n=30) | 33% (n=15) | 60% (n=15) |
| C1 v2: steps + grounded answers [lance] | `gemma-4-E2B-it` | 67% (n=30) | 47% (n=15) | 40% (n=15) |
| C2: guardrailed agent | `gemma-4-E2B-it` | 43% (n=30) | 33% (n=15) | 53% (n=15) |
| C2 v1 (guardrails logged, not enforced) | `gemma-4-E2B-it` | 36% (n=14) | 29% (n=7) | 56% (n=9) |
| closed book | `gemma-4-E2B-it` | 7% (n=30) | 0% (n=15) | 20% (n=15) |
| D: free small agent | `gemma-4-E2B-it` | 13% (n=30) | 0% (n=15) | 27% (n=15) |
| D v1 (strict parser) | `gemma-4-E2B-it` | 7% (n=14) | 0% (n=7) | 33% (n=9) |

### All questions

| Racer | Model | n | Accuracy | F1 | All evidence found | LLM calls | Tool calls | Prompt tok | Output tok | LLM s/q | Tool s/q | $ / 1K q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: basic RAG | `gemma-4-E2B-it` | 200 | **32.5%** | 0.35 | 48% | 1.0 | 1.0 | 654 | 35 | 2.5 | 0.0 | 0.244 |
| B: big agent | `gpt-oss-120b` | 31 | **77.4%** | 0.78 | 65% | 3.77 | 2.42 | 3979 | 182 | 19.9 | 5.6 | 0.733 |
| C1: fixed steps | `gemma-4-E2B-it` | 200 | **62.5%** | 0.61 | 65% | 3.5 | 1.75 | 2555 | 192 | 14.0 | 1.0 | 1.362 |
| C1: fixed steps [lance] | `gemma-4-E2B-it` | 200 | **61.0%** | 0.60 | 66% | 3.52 | 1.75 | 2607 | 193 | 14.3 | 1.1 | 1.389 |
| C1 v2: steps + grounded answers [lance] | `gemma-4-E2B-it` | 200 | **58.5%** | 0.61 | 70% | 4.33 | 2.23 | 3824 | 306 | 22.4 | 1.4 | 2.179 |
| C2: guardrailed agent | `gemma-4-E2B-it` | 200 | **45.0%** | 0.48 | 58% | 3.52 | 1.61 | 2655 | 299 | 21.6 | 0.3 | 2.095 |
| C2 v1 (guardrails logged, not enforced) | `gemma-4-E2B-it` | 101 | **35.6%** | 0.39 | 47% | 2.17 | 1.11 | 1308 | 182 | 12.9 | 0.2 | 1.258 |
| closed book | `gemma-4-E2B-it` | 200 | **2.5%** | 0.04 | - | 1.0 | 0.0 | 74 | 17 | 1.3 | 0.0 | 0.122 |
| D: free small agent | `gemma-4-E2B-it` | 200 | **20.0%** | 0.25 | 65% | 2.89 | 1.89 | 2567 | 312 | 22.2 | 0.1 | 2.162 |
| D v1 (strict parser) | `gemma-4-E2B-it` | 101 | **13.9%** | 0.14 | 27% | 1.67 | 0.67 | 1002 | 164 | 11.7 | 0.0 | 1.136 |

Accuracy by question type (all):

| Racer | Model | bridge | bridge3 | comparison |
|---|---|---|---|---|
| A: basic RAG | `gemma-4-E2B-it` | 31% (n=110) | 30% (n=50) | 40% (n=40) |
| B: big agent | `gpt-oss-120b` | 75% (n=16) | 67% (n=9) | 100% (n=6) |
| C1: fixed steps | `gemma-4-E2B-it` | 65% (n=110) | 50% (n=50) | 72% (n=40) |
| C1: fixed steps [lance] | `gemma-4-E2B-it` | 65% (n=110) | 42% (n=50) | 72% (n=40) |
| C1 v2: steps + grounded answers [lance] | `gemma-4-E2B-it` | 65% (n=110) | 52% (n=50) | 50% (n=40) |
| C2: guardrailed agent | `gemma-4-E2B-it` | 45% (n=110) | 34% (n=50) | 60% (n=40) |
| C2 v1 (guardrails logged, not enforced) | `gemma-4-E2B-it` | 35% (n=54) | 26% (n=23) | 46% (n=24) |
| closed book | `gemma-4-E2B-it` | 2% (n=110) | 0% (n=50) | 8% (n=40) |
| D: free small agent | `gemma-4-E2B-it` | 19% (n=110) | 2% (n=50) | 45% (n=40) |
| D v1 (strict parser) | `gemma-4-E2B-it` | 6% (n=54) | 0% (n=23) | 46% (n=24) |
