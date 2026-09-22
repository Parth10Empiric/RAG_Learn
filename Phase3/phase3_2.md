# Phase 3.2 — Score Distribution and Threshold Tuning

**Date:** 2026-09-22
**Branch:** `phase-3-industrial-rag`
**Scripts:** `evaluation/inspect_scores.py`, `evaluation/evaluate_thresholds.py`
**Retriever:** Phase 3 local (`RAG_Project_Industrial/retriever.py`, now supports `score_threshold`)

```bash
cd ~/Projects/RAG_Learn/Phase3
python -m RAG_Project_Industrial.evaluation.inspect_scores
python -m RAG_Project_Industrial.evaluation.evaluate_thresholds
```

**The question this phase asks:** can we add a similarity cutoff so the system says
*"I don't know"* instead of returning junk when the answer isn't in the corpus?

**The answer: no — not at this chunk size.** Details below.

---

## 1. What a score threshold is meant to do

Every retrieved chunk comes with a cosine similarity score between 0 and 1. Higher = the
chunk is more similar to the question.

The hope is that scores split cleanly into two groups:

```
   junk                                          real answers
   ├────────────┤                               ├────────────┤
 0.0          0.45          ^                 0.55          1.0
                        threshold
```

If that were true, you pick a cutoff, drop everything below it, and the system naturally
refuses to answer questions the corpus can't support. Phase 3.2 tests whether the split
actually exists.

---

## 2. What the raw scores look like

`inspect_scores.py` prints the top 20 chunks and scores for all 12 questions. The shape of
each list matters more than the individual numbers.

### Pattern 1 — a good question has one spike, then a plateau

```
q002: What is dependency injection in FastAPI?
  Rank 1  0.7960  chunk 38   <- the answer
  Rank 2  0.6120  chunk 37   <- still the answer
  Rank 3  0.5816  chunk 6    <- unrelated (application object)
  Rank 4  0.5619  chunk 1    <- unrelated (overview)
  ...
  Rank 20 0.4579  chunk 29   <- unrelated
```

There is a **0.18 cliff** between rank 1 and rank 2, then everything from rank 3 to rank 20
sits in a narrow 0.58 → 0.46 band. That band is generic FastAPI prose that looks vaguely
like every FastAPI question. Only the top 1–2 results carry real signal.

### Pattern 2 — some questions have no spike at all

```
q005: What is FastAPI middleware?
  Rank 1  0.6807  chunk 1    <- generic overview, NOT about middleware
  Rank 2  0.6420  chunk 6    <- application object, NOT about middleware
  ...
  Rank 6  0.5913  chunk 58   <- first genuinely relevant chunk
  ...
  Rank 20 0.4893  chunk 56   <- the actual definition of middleware, dead last
```

The chunk that literally defines middleware ranks **20th**. Five generic chunks outrank it.
There is no cliff — the whole list is a smooth slide from 0.68 to 0.49.

### Pattern 3 — the unanswerable questions

```
q012 (Kubernetes): top score 0.2930, flat slide to 0.2418   <- clearly nothing
q009 (Compose):    top score 0.4629, flat slide to 0.3962   <- looks like a weak answer
```

q012 is obviously empty. **q009 is not** — 0.46 is the same score a mediocre real answer
gets. This is the whole problem, and section 4 shows why.

---

## 3. The threshold sweep

`evaluate_thresholds.py` re-runs all 12 questions at `top_k=10` for each cutoff.

> `precision` here is `relevant / returned` (not `relevant / 10`), so it rises as the
> threshold shrinks the result set. `returned` is the average number of chunks surviving.

| threshold | returned | hit | precision | recall | what changed |
|---:|---:|---:|---:|---:|---|
| NONE | 10.00 | 0.7500 | 0.1333 | 0.1979 | baseline |
| 0.15 | 10.00 | 0.7500 | 0.1333 | 0.1979 | nothing |
| 0.20 | 10.00 | 0.7500 | 0.1333 | 0.1979 | nothing |
| 0.25 | 10.00 | 0.7500 | 0.1333 | 0.1979 | nothing |
| 0.30 | 9.17 | 0.7500 | 0.1333 | 0.1979 | **q012 now returns 0** ✅ |
| 0.35 | 9.00 | 0.7500 | 0.1354 | 0.1979 | trims 2 junk chunks off q011 |
| 0.40 | 8.92 | 0.6667 | 0.1250 | 0.1860 | **q011 loses its only real hit** ❌ |
| 0.45 | 7.75 | 0.6667 | 0.1250 | 0.1860 | q009 cut to 2 chunks — still answering |
| 0.50 | 7.08 | 0.6667 | 0.1167 | 0.1709 | **q009 finally returns 0** ✅ but q011 is dead |
| 0.60 | 2.83 | 0.5000 | **0.2181** | 0.1195 | 5 of 12 questions return nothing |

### Reading it in plain English

- **0.15 → 0.25 does literally nothing.** No chunk in any top-10 scores that low, so the
  filter never fires. These rows are wasted settings.
- **0.30 is the first useful cutoff.** The Kubernetes question now correctly returns zero
  chunks. Note that **no metric improved** — see section 5 for why that's a measurement bug,
  not good news.
- **0.40 is where it breaks.** q011 ("create a Python API and package it into a container")
  has exactly one relevant chunk in its top 10, at rank 8 with score **0.3913**. A 0.40
  cutoff deletes it. `hit` drops from 9/12 to 8/12.
- **0.50 finally silences the Compose question** — but q011 is already gone, and recall has
  fallen from 0.1979 to 0.1709.
- **0.60 looks like the best precision (0.2181) and is the worst setting on the table.**
  Precision only rises because the denominator collapsed to 2.83 chunks. Five of twelve
  questions — including two perfectly answerable ones, q006 and q008 — return nothing at
  all. This is the classic trap: *precision always improves when you refuse to answer.*

---

## 4. The central finding: the two groups overlap

For a threshold to work, this must hold:

$$
\min_{\text{answerable } q}\big(\text{score of first \textbf{relevant} chunk}\big)
\;>\;
\max_{\text{unanswerable } q}\big(\text{top score}\big)
$$

Measured directly:

| | score | query |
|---|---:|---|
| lowest "first relevant" score, answerable | **0.3913** | q011 |
| highest score on an unanswerable question | **0.4629** | q009 |
| **usable separation** | **−0.0716** | — |

**The gap is negative.** The best *wrong* answer scores higher than the weakest *right*
answer. No single number can sit between them. Any cutoff that rejects Docker Compose also
rejects the multi-hop question — which the sweep confirms: q011 dies at 0.40, q009 survives
until 0.50.

### ⚠️ Correction to Phase 3.1

`phase 3_1.md` §5 reported a **positive** separation margin of +0.017 (MiniLM) and +0.026
(BGE-M3), and concluded a threshold was possible but fragile. **That was wrong.**

It compared the *top-1 score* of each query without checking whether the rank-1 chunk was
actually relevant. For q011 the rank-1 chunk scores 0.4794 — but it is `fastapi.txt` chunk
70 (APIRouter), which is **not** in the gold set. The first genuinely relevant chunk is at
rank 8, scoring 0.3913.

Measuring the score of a wrong answer made the gap look positive. Corrected:

```
phase 3_1.md  (top-1 score, relevance ignored):   +0.017   "fragile but possible"
phase 3_2.md  (first RELEVANT score):             −0.0716  "impossible"
```

### Where rank 1 is not even relevant

| query | top-1 score | rank-1 relevant? | first relevant | at rank |
|---|---:|:---:|---:|---:|
| q001 | 0.7749 | ✅ | 0.7749 | 1 |
| q002 | 0.7960 | ✅ | 0.7960 | 1 |
| q003 | 0.7464 | ✅ | 0.7464 | 1 |
| q004 | 0.7425 | ✅ | 0.7425 | 1 |
| q005 | 0.6807 | ❌ | 0.5913 | 6 |
| q006 | 0.5662 | ❌ | 0.5566 | 2 |
| q007 | 0.7436 | ✅ | 0.7436 | 1 |
| q008 | 0.5885 | ❌ | **none in top 50** | — |
| q010 | 0.6578 | ❌ | 0.6004 | 4 |
| q011 | 0.4794 | ❌ | 0.3913 | 8 |

Half the answerable questions lead with an irrelevant chunk. **q008 has no relevant chunk
anywhere in the top 50** — it scores a confident 0.5885 on something wrong, which no
threshold can catch, because the model is not uncertain. It is wrong.

---

## 5. A measurement bug that hides the good news

At threshold 0.30 the Kubernetes question correctly returned **zero chunks** — exactly the
behaviour we want. Yet every metric stayed identical.

Why: `evaluate_thresholds.py` averages over all 12 questions, and for q009/q012 the gold set
is empty, so `hit`, `precision` and `recall` are all 0.0 **whether the system abstains
correctly or returns ten pieces of garbage**. Correct refusal earns nothing.

This is the same flaw already logged against `compare_embeddings.py` in `phase 3_1.md` §3.
Consequences here:

- the sweep cannot show that 0.30 fixed q012;
- all numbers are scaled by 10/12, so `recall 0.1979` is really **0.2375** over answerable
  questions;
- there is no metric anywhere that rewards saying "I don't know".

**Needed:** split the report into two numbers — retrieval quality over the 10 answerable
questions, and a separate *abstention rate* over the 2 unanswerable ones.

---

## 6. Conclusions

1. **A score threshold cannot work on this corpus at this chunk size.** Usable separation is
   −0.0716. The band between "rejects Compose" (≥0.50) and "keeps the multi-hop answer"
   (≤0.39) is empty.

2. **0.30 is the only safe setting today.** It removes the obviously-empty Kubernetes case
   and costs nothing (`hit`, `precision`, `recall` all unchanged). It is a cheap guard
   against wildly off-topic questions, not a real "I don't know" mechanism.

3. **Never tune on precision alone.** 0.60 gives the best precision on the table (0.2181)
   while silently refusing 5 of 12 questions. Precision and abstention rate must be read
   together.

4. **The root cause is chunking, not thresholding.** Scores cluster in a narrow 0.45–0.60
   band because ~200-character chunks are too small to be distinctive — generic FastAPI
   prose scores nearly as high as the actual answer. Phase 3.1 reached the same conclusion
   from recall (0.238 vs a 0.958 ceiling). Fixing chunk size should widen the score
   distribution and may make thresholding viable for the first time.

5. **Two questions are badly served and should be revisited:** q008 (no relevant chunk in
   the top 50 despite a confident 0.5885) and q009 (labelled answerable from `docker.pdf`,
   but Compose is absent from the corpus).

---
