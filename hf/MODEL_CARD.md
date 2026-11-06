---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- retrieval
- education
language:
- en
license: apache-2.0
base_model: BAAI/bge-base-en-v1.5
---

# edu-v1

edu-v1 is an English sentence embedding model. It is used to match short descriptions of student interests, such as "biology but with computers", to passages about undergraduate programs. It is a fine-tuned version of [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It was built for an admissions guidance product for IB students and is the default model of the edu-embedding retrieval pipeline.

## Results at a glance

| Test | edu-v1 | Base model bge-base-en-v1.5 | Best other model |
|---|---:|---:|---:|
| Retrieve a program from another section of its description, Recall@10 (3,695 queries) | **67.0%** | 45.1% | 54.5% (all-MiniLM-L6-v2) |
| Triplet accuracy, private validation split 2 | **98.1%** | 94.8% | 92.7% (bge-small-en-v1.5) |
| Student-style queries, share of top-10 programs in an acceptable field (40 queries) | 91.8% | **94.8%** | 94.3% (bge-small-en-v1.5) |
| Mean cosine of random corpus passages (lower is more spread out) | **0.026** | 0.615 | 0.352 (all-MiniLM-L6-v2) |

edu-v1 is clearly better on the first two tests, but both use text from its own training data (see Limitations). On short student-style queries it is not better than its base model.

## Model details

| | |
|---|---|
| Architecture | BERT base, 12 layers, hidden size 768, about 110M parameters |
| Output | 768-dimensional vector, CLS pooling, L2-normalised |
| Similarity | Cosine |
| Maximum input | 512 tokens, lowercased |
| Language | English |
| Base model | BAAI/bge-base-en-v1.5 (measured, see below) |

The card does not name a base model, so we compared the weights directly. The embedding layer and encoder layers 0 to 7 are bit-identical to BAAI/bge-base-en-v1.5. Layers 8 to 11 differ by 1.1 to 1.6% (relative Frobenius norm), and the whole model by 0.6%. This fits a fine-tune of the top four layers only. It also means edu-v1 stays very close to its base model.

## Intended use

Ranking passages about university programs against short English queries written by students. It works best inside a retrieval pipeline that also uses keyword search. See the edu-embedding repository for the pipeline. It was built on a private corpus of undergraduate programs at universities in the UK, Hong Kong and Singapore.

Not intended for:

1. Admissions decisions or advice without a person checking the results.
2. Classification with a fixed similarity threshold. Scores are only meaningful for ranking.
3. Facts about programs. The model ranks text; it does not check it.
4. Languages other than English.

## Training (as described by the author)

No training code or training log has been published. The previous version of this card described the training as follows. None of it has been verified.

| Item | Author's description |
|---|---|
| Data | Triplets of a query, a matching program passage and a non-matching one |
| Negatives | Three levels: different cluster, same cluster, and closely related programs |
| Loss | TripletLoss (weight 0.7), an alignment loss toward a larger teacher model (0.2), and a variance loss against collapse (0.1) |

The fine-tuning triplets are private. Two candidate triplet sets exist, both generated from the program corpus with templated queries, and it is not recorded which one trained edu-v1. They contain 97.7% and 99.6% of the corpus passages. The queries are template-generated, not collected from real students.

## Evaluation

All numbers come from the edu-embedding repository (`edu-embedding eval`), on Apple MPS with seed 0 and pinned revisions. Baselines: BAAI/bge-base-en-v1.5, BAAI/bge-small-en-v1.5 and sentence-transformers/all-MiniLM-L6-v2. The corpus and the triplet splits are private.

**A. Triplet accuracy.** Share of validation triplets where cos(anchor, positive) is higher than cos(anchor, negative).

| Model | Private validation split 1 | Private validation split 2 | Mean cos(a,p) | Mean cos(a,n) | Mean margin |
|---|---:|---:|---:|---:|---:|
| edu-v1 | 97.2% | 98.1% | 0.350 | 0.001 | 0.349 |
| bge-base-en-v1.5 | 96.3% | 94.8% | 0.716 | 0.606 | 0.109 |
| bge-small-en-v1.5 | 94.4% | 92.7% | 0.782 | 0.695 | 0.088 |
| all-MiniLM-L6-v2 | 93.2% | 91.1% | 0.562 | 0.369 | 0.194 |

Cosine columns are for split 2. The previous card reported a positive similarity of about 0.38, a negative similarity of about 0.03 and a margin above 0.35. Those figures are close to the ones measured here on this split. Every positive passage in both validation splits also appears in the training splits.

**B. Cross-section program retrieval.** Each passage of the 739 programs with exactly five sections is a query, with tags and university and program names removed. The target is the same program; its other four passages stay in the index. Programs are ranked by their best passage.

| Model | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| edu-v1 | 18.9% | 50.1% | 67.0% | 0.323 |
| all-MiniLM-L6-v2 | 13.0% | 39.9% | 54.5% | 0.243 |
| BM25, no model | 15.8% | 38.6% | 48.9% | 0.253 |
| bge-base-en-v1.5 | 9.3% | 30.5% | 45.1% | 0.185 |
| bge-small-en-v1.5 | 9.5% | 30.0% | 44.7% | 0.185 |

**C. Student-interest queries.** 40 short queries such as "how the brain works", each labelled by the repository author with the acceptable fields (14 coarse clusters). Dense search only, plain query. P@10 is the share of the top 10 programs in an acceptable field.

| Model | P@10 | 95% bootstrap interval |
|---|---:|---:|
| bge-base-en-v1.5 | 94.8% | 92.0 to 97.2 |
| bge-small-en-v1.5 | 94.3% | 91.7 to 96.8 |
| all-MiniLM-L6-v2 | 93.3% | 89.5 to 96.5 |
| edu-v1 | 91.8% | 87.8 to 95.2 |

With the production query template, edu-v1 reaches 89.2% and bge-base-en-v1.5 86.8%.

**D. Embedding space.** On all corpus passages, the mean cosine of random pairs is 0.026 for edu-v1 (bge-base-en-v1.5: 0.615), and the first principal component holds 3.9% of the variance (bge-base-en-v1.5: 5.6%). The space has not collapsed.

## Limitations

1. The training triplets were built from the same corpus that the retrieval evaluation uses; they contain up to 99.6% of its passages. Parts B and C below therefore test edu-v1 on passages it has very likely seen in training. The baselines have not seen them.
2. The 40 student-interest queries were written and labelled by the repository author, at the level of 14 coarse clusters. They are a sanity check, not a benchmark.
3. The program corpus has unknown provenance and has not been checked against university pages.
4. The model is English only and was tuned on UK, Hong Kong and Singapore programs.

## How to use

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ongilLabs/edu-v1")

query = "biology but with computers"
passages = [
    "Bioinformatics combines molecular biology with programming and statistics.",
    "A degree in English literature covering poetry, drama and the novel.",
]
q = model.encode(query, normalize_embeddings=True)
p = model.encode(passages, normalize_embeddings=True)
print(p @ q)  # higher means a better match
```

The production pipeline wrapped queries in a fixed template before encoding:

```python
query = "[CLUSTER:life-sciences][SECTION:overview] Recommend undergraduate majors matching: biology but with computers."
```

The training data used this template too. In our tests the plain query worked about as well or better (see part B and C). The full pipeline, with BM25, filters and diversification, is in the edu-embedding repository.

## License

Apache 2.0, as declared on the previous card. The base model, BAAI/bge-base-en-v1.5, is MIT licensed; keep its copyright notice when redistributing.
