# Corpus format

The program corpus used to build and evaluate edu-v1 is private and is not part of this repository. Neither are the triplets used for fine-tuning. The corpus covers undergraduate programs at universities in the UK, Hong Kong and Singapore. Its text reads like summaries of public program pages, may be out of date, and has not been checked against those pages.

To use the retrieval pipeline, bring your own corpus in the format below and pass it with `--corpus PATH` or the `EDU_EMBED_CORPUS` environment variable. There is no default path. Without a corpus, `edu-embedding search` and `edu-embedding eval` stop with an error.

## One record per line

Each line is one JSON object: one section of one program.

```json
{
  "id": "example_university_data_science_bsc_overview_001",
  "title": "Example University | Data Science BSc | Overview",
  "text": "[UNIV:Example University][MAJOR:Data Science BSc][LANG:EN][SECTION:overview] A three-year degree in statistics, programming and machine learning, with a final-year project on real data.",
  "meta": {
    "university": "Example University",
    "program_name": "Data Science BSc",
    "section": "overview",
    "language": "en",
    "cluster": "data-science"
  }
}
```

| Field | Rule |
|---|---|
| `id` | Non-empty string, unique across the file. The loader rejects duplicates. |
| `title` | Short label for display. The lexical guard and the cross-encoder also read it. |
| `text` | The passage. Production texts start with the four tags shown above, followed by the body. The tags are optional; when present they are stripped for display and for part B queries. |
| `meta.university` | University name. Passages with the same university and program name form one program. |
| `meta.program_name` | Program name. The taxonomy filter and the part C labels read it. |
| `meta.section` | One of `overview`, `core_modules`, `skills`, `capstone`, `outcomes`. Programs can have fewer than five sections. |
| `meta.language` | Language code, for example `en`. The models are English only. |
| `meta.cluster` | One of the 14 clusters in `src/edu_embedding/taxonomy.py`, or `unknown`. Only the lexical guard reads it. |

Section weights used when passage scores are summed into a program score: overview 1.25, core modules 1.2, skills 1.1, capstone 0.95, outcomes 0.8.

## Checking and cleaning your own file

```python
from edu_embedding.corpus import dedupe_ids, load_corpus, read_records

records = read_records("programs.jsonl")
records, report = dedupe_ids(records)   # drop exact repeats, rename clashing ids
passages = load_corpus("programs.jsonl")  # validates every record, raises on errors
```

`load_corpus` checks the required keys, the section names and id uniqueness. `dedupe_ids` drops a record whose text equals an earlier record with the same id, and renames the others with a `__2`, `__3` suffix.

## How the private corpus was prepared

The private corpus went through the same id clean-up, plus a few corpus-specific fixes: trimming spaces around names, merging two spellings of one university, and removing citation markers left by a chat assistant. Part B of the evaluation needs programs with exactly one passage per section, so a corpus without the five sections can still be searched but gives fewer part B queries.
