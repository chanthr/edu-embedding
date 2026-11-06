### A. Triplet accuracy (private validation splits)

| Model | Split | Accuracy % | Mean cos(a,p) | Mean cos(a,n) | Mean margin |
|---|---:|---:|---:|---:|---:|
| edu-v1 | private split 1 | 97.2 | 0.478 | 0.056 | 0.422 |
| edu-v1 | private split 2 | 98.1 | 0.350 | 0.001 | 0.349 |
| bge-base-en-v1.5 | private split 1 | 96.3 | 0.783 | 0.624 | 0.158 |
| bge-base-en-v1.5 | private split 2 | 94.8 | 0.716 | 0.606 | 0.109 |
| bge-small-en-v1.5 | private split 1 | 94.4 | 0.820 | 0.704 | 0.117 |
| bge-small-en-v1.5 | private split 2 | 92.7 | 0.782 | 0.695 | 0.088 |
| all-MiniLM-L6-v2 | private split 1 | 93.2 | 0.644 | 0.407 | 0.236 |
| all-MiniLM-L6-v2 | private split 2 | 91.1 | 0.562 | 0.369 | 0.194 |

### A. Overlap with the matching training triplets

| Split | Corpus passages seen in training % | Validation positives seen in training % | Validation anchors in training % | Validation triplets in training % |
|---|---:|---:|---:|---:|
| private split 1 | 97.7 | 100.0 | 51.5 | 4.2 |
| private split 2 | 99.6 | 99.5 | 6.4 | 0.0 |

### B. Cross-section program retrieval (3695 queries)

| Model | Query format | R@1 % | R@5 % | R@10 % | MRR@10 |
|---|---:|---:|---:|---:|---:|
| edu-v1 | plain | 18.9 | 50.1 | 67.0 | 0.323 |
| all-MiniLM-L6-v2 | plain | 13.0 | 39.9 | 54.5 | 0.243 |
| BM25 (no model) | plain | 15.8 | 38.6 | 48.9 | 0.253 |
| bge-base-en-v1.5 | plain | 9.3 | 30.5 | 45.1 | 0.185 |
| bge-small-en-v1.5 | plain | 9.5 | 30.0 | 44.7 | 0.185 |
| bge-base-en-v1.5 +instr | plain | 8.6 | 27.6 | 41.5 | 0.170 |
| edu-v1 | template | 20.3 | 50.9 | 67.2 | 0.334 |
| all-MiniLM-L6-v2 | template | 14.9 | 40.9 | 54.1 | 0.258 |
| bge-small-en-v1.5 | template | 9.6 | 32.4 | 48.5 | 0.196 |
| bge-base-en-v1.5 | template | 10.1 | 33.3 | 48.5 | 0.201 |
| bge-base-en-v1.5 +instr | template | 10.0 | 31.6 | 45.8 | 0.194 |

### C. Student-interest queries (40 queries, author-labelled)

| Model | Setting | P@10 % | 95% interval | Precision of returned % | Programs returned | Unique universities |
|---|---:|---:|---:|---:|---:|---:|
| edu-v1 | dense, plain query | 91.8 | 87.8 to 95.2 | 91.8 | 10.0 | 6.6 |
| edu-v1 | dense, production template | 89.2 | 84.0 to 93.5 | 89.2 | 10.0 | 6.8 |
| edu-v1 | full recommender | 69.8 | 58.0 to 81.0 | 92.4 | 7.5 | 5.4 |
| bge-base-en-v1.5 | dense, plain query | 94.8 | 92.0 to 97.2 | 94.8 | 10.0 | 6.4 |
| bge-base-en-v1.5 | dense, production template | 86.8 | 81.0 to 91.8 | 86.8 | 10.0 | 6.2 |
| bge-base-en-v1.5 | full recommender | 71.0 | 59.2 to 82.0 | 94.4 | 7.6 | 5.4 |
| bge-base-en-v1.5 +instr | dense, plain query | 94.3 | 91.0 to 97.0 | 94.3 | 10.0 | 6.7 |
| bge-base-en-v1.5 +instr | dense, production template | 88.0 | 83.3 to 92.2 | 88.0 | 10.0 | 6.6 |
| bge-base-en-v1.5 +instr | full recommender | 71.5 | 60.2 to 82.2 | 94.3 | 7.7 | 5.5 |
| bge-small-en-v1.5 | dense, plain query | 94.3 | 91.7 to 96.8 | 94.3 | 10.0 | 6.4 |
| bge-small-en-v1.5 | dense, production template | 90.3 | 84.2 to 95.0 | 90.3 | 10.0 | 5.9 |
| bge-small-en-v1.5 | full recommender | 72.8 | 61.5 to 83.0 | 97.3 | 7.5 | 5.5 |
| all-MiniLM-L6-v2 | dense, plain query | 93.3 | 89.5 to 96.5 | 93.3 | 10.0 | 6.3 |
| all-MiniLM-L6-v2 | dense, production template | 89.2 | 83.5 to 94.0 | 89.2 | 10.0 | 6.2 |
| all-MiniLM-L6-v2 | full recommender | 71.5 | 60.2 to 82.0 | 94.8 | 7.7 | 5.3 |

### D. Embedding space health (corpus passages)

| Embeddings | Dim | Mean cos, random pairs | Std | Variance on 1st PC % |
|---|---:|---:|---:|---:|
| edu-v1 | 768 | 0.026 | 0.114 | 3.9 |
| bge-base-en-v1.5 | 768 | 0.615 | 0.060 | 5.6 |
| bge-small-en-v1.5 | 384 | 0.709 | 0.058 | 6.4 |
| all-MiniLM-L6-v2 | 384 | 0.352 | 0.129 | 6.7 |
| Saved production embeddings (old repository, not included) | 768 | 0.470 | 0.881 | 99.8 |

### E. Recommender ablation on the student-interest queries

| Step | edu-v1 P@10 % | edu-v1 precision of returned % | edu-v1 returned | bge-base-en-v1.5 P@10 % | bge-base-en-v1.5 precision of returned % | bge-base-en-v1.5 returned |
|---|---:|---:|---:|---:|---:|---:|
| Dense only | 90.8 | 90.8 | 10.0 | 89.2 | 89.2 | 10.0 |
| BM25 only | 77.3 | 83.9 | 9.1 | 77.3 | 83.9 | 9.1 |
| Hybrid (dense + BM25) | 91.5 | 91.5 | 10.0 | 91.8 | 91.8 | 10.0 |
| + lexical guard and taxonomy filter | 69.8 | 92.4 | 7.5 | 71.2 | 94.7 | 7.6 |
| + per-university cap and MMR | 69.8 | 92.4 | 7.5 | 71.0 | 94.4 | 7.6 |
| + cross-encoder rerank (production defaults) | 69.8 | 92.4 | 7.5 | 71.0 | 94.4 | 7.6 |

### P. Weight comparison with the base model

| Pair | Overall relative difference % | Identical blocks | Changed blocks |
|---|---:|---:|---:|
| edu-v1 vs bge-base-en-v1.5 | 0.6 | 13 of 17 | encoder.layer.08, encoder.layer.09, encoder.layer.10, encoder.layer.11 |
