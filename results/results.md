# Results

Model claude-haiku-5-5. 30 planted facts, each asked by an employee allowed to read its document and by one who is not.
Every configuration ran 3 times. A cell gives the lowest and the highest count over those runs.

| Method | Filter forgotten | Answered (of 30) | Fact reached the model (of 30) | Fact in the answer (of 30) | Existence disclosed (of 30) | Title words named (of 30) | Cost (USD) |
|---|---|---|---|---|---|---|---|
| rls | no | 30 | 0 | 0 | 3 to 5 | 0 to 1 | 0.0564 |
| rls | yes | 30 | 0 | 0 | 4 to 6 | 0 to 2 | 0.0587 |
| in_query | no | 30 | 0 | 0 | 2 to 4 | 0 to 1 | 0.0567 |
| in_query | yes | 30 | 30 | 30 | 30 | 16 to 17 | 0.0543 |
| post_filter | no | 30 | 0 | 0 | 4 to 5 | 1 to 2 | 0.0568 |
| post_filter | yes | 30 | 30 | 30 | 30 | 16 to 17 | 0.0547 |
| prompt_only | no | 30 | 30 | 0 to 1 | 0 to 1 | 0 to 1 | 0.0573 |

Answered counts allowed employees whose answer contains the fact. The other counts are for denied employees.
Fact reached the model: a search result shown to the model contains the fact.
Fact in the answer: the answer contains the fact.
Existence disclosed: the answer is not the fixed refusal.
Title words named: the answer contains a word of the document title that the question did not contain.

Total: 2,918,883 input and 205,934 output tokens, 0.3949 USD.

## Retrieval rank

The rank of the fact's document in the allowed employee's search for the question, k = 5. The last column counts denied employees whose search returns it.

| Method | Filter forgotten | Top 1 | Top 3 | Top 5 | Returned to the denied employee |
|---|---|---|---|---|---|
| rls | no | 29 | 30 | 30 | 0 |
| rls | yes | 29 | 30 | 30 | 0 |
| in_query | no | 29 | 30 | 30 | 0 |
| in_query | yes | 29 | 30 | 30 | 30 |
| post_filter | no | 29 | 30 | 30 | 0 |
| post_filter | yes | 29 | 30 | 30 | 30 |
| prompt_only | no | 29 | 30 | 30 | 30 |

## Answers lost by post_filter

post_filter reads k x 4 chunks and then drops those the employee may not read. A lost answer is a fact document that rls returns to the allowed employee within k hits and post_filter does not. No model is involved.

| k | Found by rls | Found by post_filter | Lost |
|---|---|---|---|
| 1 | 29 | 29 | 0 |
| 3 | 30 | 30 | 0 |
| 5 | 30 | 30 | 0 |

## Reproduce

```
docker compose up -d --wait db
PYTHONPATH=src .venv/bin/python -m needtoknow.ingest
PYTHONPATH=src .venv/bin/python -m needtoknow.evaluate --out results/
```
