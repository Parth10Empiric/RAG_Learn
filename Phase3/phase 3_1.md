# Phase 3.1 — Embedding Model Comparison

**Date:** 2026-09-21
**Branch:** `phase-3-industrial-rag`
**Script:** `Phase3/RAG_Project_Industrial/evaluation/compare_embeddings.py`

Goal: decide whether to keep `all-MiniLM-L6-v2` from Phase 2 or move to `BAAI/bge-m3`,
measured against a hand-labelled ground truth rather than by eyeballing results.

---

## 1. Setup

| | |
|---|---|
| Corpus | 323 chunks in Qdrant (`developer_knowledge_v1`) |
| Sources | `fastapi.txt` (136 chunks), `docker.pdf` (187 chunks) |
| Questions | 12 (`evaluation/retrieval_questions.json`) |
| Ground truth | 68 judgments (`evaluation/qrels.json`) |
| Models | `all-MiniLM-L6-v2` (384-dim) vs `BAAI/bge-m3` (1024-dim) |

The comparison re-embeds all 323 chunks in memory with each model and ranks them by
cosine similarity. It does **not** write to Qdrant, so BGE-M3 can be evaluated without
touching the live 384-dim collection.

### Ground truth

Relevance was judged against **the corpus**, not against what the retriever returned.
This matters: the source documents are split into ~200-char (fastapi) and ~500-char
(docker) chunks, so every topic spans a *run* of consecutive chunks, and only the chunk
containing the section header scores highly. Labelling only what was retrieved would
have hidden exactly the failure the evaluation exists to find.

Two questions have an empty gold set because the answer is not in the corpus at all
(zero occurrences of "Compose" or "Kubernetes" in either document):

- **q009** — "What is Docker Compose used for?" — unintended gap; `docker.pdf` is only
  Chapter 1 of Turnbull's *Docker Book* (images and repositories).
- **q012** — "How does Kubernetes schedule pods?" — deliberate negative control.

---

## 2. The metrics, and how to read them

### Notation

| Symbol | Meaning |
|---|---|
| $Q$ | the set of evaluation queries |
| $R_q$ | the **gold set** — every chunk judged relevant to query $q$ in `qrels.json` |
| $L_q$ | the full ranked list of chunks returned for $q$, best first |
| $L_q^{(k)}$ | the first $k$ entries of $L_q$ |
| $h_q(k) = \lvert L_q^{(k)} \cap R_q \rvert$ | how many relevant chunks landed in the top $k$ |

Every metric below is computed per query, then averaged over $Q$.

### Hit@k — "did we get *anything* useful?"

$$
\text{Hit@}k(q) = \begin{cases} 1 & h_q(k) \ge 1 \\ 0 & \text{otherwise} \end{cases}
\qquad\qquad
\text{Hit@}k = \frac{1}{|Q|}\sum_{q \in Q} \text{Hit@}k(q)
$$

Binary per query. It ignores *how many* relevant chunks came back and *where* they ranked.
Hit@10 = 0.90 means "for 9 of 10 questions, at least one relevant chunk is somewhere in
the top 10" — it says nothing about the other nine slots.

### Precision@k — "how much of what we returned was useful?"

$$
\text{P@}k(q) = \frac{h_q(k)}{k}
$$

The denominator is $k$, **not** the number of relevant chunks. This is the metric that
matters for context-window cost: P@10 = 0.18 means 8.2 of every 10 chunks you stuff into
the prompt are noise.

> ⚠️ **P@k is capped when $\lvert R_q \rvert < k$.** If only 3 chunks in the whole corpus
> are relevant (q005, q008), then $h_q(10) \le 3$ and P@10 can never exceed 0.30 no matter
> how perfect the retriever is. See *Ceilings* below.

### Recall@k — "how much of the answer did we find?"

$$
\text{R@}k(q) = \frac{h_q(k)}{\lvert R_q \rvert}
$$

The denominator is the gold set size. This is the metric that matters for **answer
completeness**: if the answer spans 8 chunks and you retrieve 4, the LLM is reasoning from
half the evidence. Recall is the metric this project is currently failing.

### MRR — "how high was the *first* useful result?"

$$
\text{RR}(q) = \frac{1}{\operatorname{rank}_q}
\qquad\qquad
\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\operatorname{rank}_q}
$$

where $\operatorname{rank}_q$ is the position of the **first** relevant chunk in $L_q$
(and $\text{RR}(q) = 0$ if none is found). Only the first hit counts — everything after it
is invisible to MRR.

$$
\operatorname{rank}_q = 1 \Rightarrow 1.00 \qquad
2 \Rightarrow 0.50 \qquad
3 \Rightarrow 0.33 \qquad
5 \Rightarrow 0.20 \qquad
10 \Rightarrow 0.10
$$

The drop-off is steep: rank 1 is worth twice rank 2. This is why MRR is the best proxy for
perceived quality — the top chunk dominates the generated answer.

### Worked example — q001, MiniLM, $k = 10$

Question: *"How does FastAPI validate request bodies?"*
Gold set: `fastapi.txt` chunks {22, 23, 24, 25, 26, 27, 28, 29}, so $\lvert R_q \rvert = 8$.
Top 10 returned: 25, 29, 2, 23, 63, 22, 119, 1, 101, 94 — of which **25, 29, 23, 22** are
in the gold set, so $h_q(10) = 4$, and the first hit is at rank 1.

$$
\text{Hit@}10 = 1
\qquad
\text{P@}10 = \frac{4}{10} = 0.40
\qquad
\text{R@}10 = \frac{4}{8} = 0.50
\qquad
\text{RR} = \frac{1}{1} = 1.00
$$

Perfect MRR, yet **half the answer was never retrieved** (chunks 24, 26, 27, 28 were missed;
of those, 24, 27 and 28 do not appear even in the top 25).
That single query is the whole report in miniature.

### Ceilings — what a perfect retriever would score here

Because several gold sets are smaller than $k = 10$, the averages cannot reach 1.0:

$$
\max \text{P@}10 = \frac{1}{|Q|}\sum_q \frac{\min(\lvert R_q \rvert, 10)}{10} = \mathbf{0.620}
\qquad
\max \text{R@}10 = \frac{1}{|Q|}\sum_q \frac{\min(10, \lvert R_q \rvert)}{\lvert R_q \rvert} = \mathbf{0.958}
$$

Gold set sizes: q001=8, q002=5, q003=5, q004=4, q005=3, q006=11, q007=7, q008=3, q010=15,
q011=7. Eight of the ten have $\lvert R_q \rvert < 10$.

**So judge the results against 0.620 and 0.958, not against 1.0.** BGE-M3's P@10 of 0.180
is 29% of achievable; its R@10 of 0.264 is 28% of achievable.

### Why the four disagree so violently in this report

| | MiniLM | BGE-M3 | ceiling |
|---|---|---|---|
| MRR | 0.606 | 0.666 | 1.000 |
| Recall@10 | 0.238 | 0.264 | 0.958 |

MRR near 0.67 with recall near 0.26 is the signature of **retrieving the right topic but
only a fragment of it**. The section-header chunk ranks first (high MRR); the continuation
chunks holding the rest of the answer are never retrieved (low recall). A single-metric
report would have missed this entirely — which is why all four are tracked.

---

## 3. Results as printed by the script

```
                 minilm    bge_m3
hit@1            0.4167    0.5000
hit@3            0.5000    0.5000
hit@5            0.5833    0.6667
hit@10           0.7500    0.7500
precision@1      0.4167    0.5000
precision@3      0.2222    0.2778
precision@5      0.1667    0.2333
precision@10     0.1333    0.1500
recall@1         0.0765    0.0841
recall@3         0.1111    0.1291
recall@5         0.1271    0.1805
recall@10        0.1979    0.2201
mrr              0.5047    0.5545
```

**These numbers are distorted.** `calculate_metrics()` returns 0.0 for every metric when
the gold set is empty, so q009 and q012 each contribute a hard zero to all 13 averages.
Consequences:

- every metric is capped at 10/12 = **0.833**;
- both models take the same unconditional penalty, so the **gap between them is reported
  at 83% of its true size**;
- a model that *correctly* retrieves nothing for the Kubernetes question scores the same
  as one returning garbage — the negative control cannot do its job.

Fix: exclude empty-gold queries from the retrieval averages and score them separately.
**Not yet applied to the script.**

## 4. Results over answerable queries only (n = 10)

```
                 ── as printed (n=12) ──    ── answerable only (n=10) ──
                 minilm   bge_m3   delta     minilm   bge_m3   delta
hit@1            0.4167   0.5000   +0.083    0.5000   0.6000   +0.100
hit@10           0.7500   0.7500    0.000    0.9000   0.9000    0.000
precision@10     0.1333   0.1500   +0.017    0.1600   0.1800   +0.020
recall@10        0.1979   0.2201   +0.022    0.2375   0.2642   +0.027
mrr              0.5047   0.5545   +0.050    0.6057   0.6655   +0.060
```

The correction is a uniform 12/10 rescaling, as expected.

`hit@10` is identical at **0.90** for both — not a quality tie but a corpus ceiling. Both
models find something relevant within 10 for 9 of the 10 answerable questions. The miss is
q008 ("What is a Docker container?"), which has no real section in this corpus.

---

## 5. Abstention behaviour

Top-1 similarity on the two questions with no answer in the corpus:

```
              q009 (Compose)   q012 (Kubernetes)
minilm            0.463             0.293
bge_m3            0.565             0.479
```

BGE-M3 scores unanswerable questions noticeably higher. The initial read — that this
indicates worse calibration — was **wrong**. It is a uniform upward shift, and the
separation margin is in fact the wider of the two:

```
                        minilm    bge_m3
lowest answerable       0.479     0.590    (both q011, the multi-hop question)
highest unanswerable    0.463     0.565    (both q009, Docker Compose)
separation margin      +0.017    +0.026
```

**Neither margin is usable.** 0.017 and 0.026 across 12 questions is noise; a threshold
placed in that gap is fitted to two data points. Two structural reasons:

- **q009 is the hard negative, not q012.** Kubernetes is easy to reject because nothing in
  the corpus is near it. "Docker Compose" sits adjacent to Docker content that *is*
  present, so it scores almost as high as a real answer. The dangerous unanswerable
  questions are the ones adjacent to covered material.
- **q011 is the weakest answerable query for both models.** It needs chunks from two
  documents and no single chunk scores well, so any threshold high enough to reject q009
  also rejects q011.

Note also that q008 scores 0.588 / 0.603 — above the unanswerable band — while its top-1
chunk is not relevant. A top-1 score threshold cannot catch that failure, because the
model is confidently retrieving the wrong thing rather than being unsure.

---

## 6. Cost

From this run, encoding all 323 chunks:

| | MiniLM | BGE-M3 |
|---|---|---|
| Encode time | 12 s (1.17 s/batch) | 4 min 31 s (24.70 s/batch) |
| Relative | 1× | **~21× slower** |
| Model size | ~90 MB | ~2.2 GB |
| Vector dim | 384 | 1024 |

`store.py` sets `VECTOR_SIZE = 384`, so adopting BGE-M3 requires a **new collection and a
full re-ingest**, not a config change. Storage grows ~2.7× and query latency rises.

---

## 7. Conclusions

1. **BGE-M3 is better on every measured axis** — hit@1 0.50 → 0.60, MRR 0.61 → 0.67,
   recall@10 0.238 → 0.264, separation margin +0.017 → +0.026. It gains most where MiniLM
   is weakest: q006 (0.566 → 0.746) and q011 (0.479 → 0.590), the broad and multi-hop
   questions.

2. **The encoder is not the bottleneck.** Recall@10 is 0.238 vs 0.264 — both terrible.
   Both models retrieve the chunk holding the section header and miss the continuation
   chunks. Swapping encoders moved recall by 0.027. **Fixing chunking is worth more than
   fixing the encoder**, and costs nothing at query time.

3. **Chunk sizes are inconsistent between sources.** `fastapi.txt` maxes at 199 chars,
   `docker.pdf` at 514 — `ingest.py` currently says `chunk_size=500, overlap=75`, so
   fastapi.txt predates that setting. This biases cosine scores across sources, which
   matters directly for q011 where both compete.

4. **Abstention is not solvable by a top-1 threshold** on this corpus at this chunk size.

## 8. Next steps

- [ ] Fix `calculate_metrics()` to skip empty gold sets; re-run to get undistorted numbers
- [ ] Re-ingest both sources at one consistent chunk size; re-run this comparison
      **before** deciding on BGE-M3 — the ranking may change once recall is not
      chunking-limited
- [ ] Review q008 and q009 — both are mislabelled in `retrieval_questions.json` as
      answerable from `docker.pdf` when the corpus does not cover them
- [ ] Revisit abstention once chunking is fixed; consider top-k mean score or a reranker
      rather than top-1 similarity
