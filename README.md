# needtoknow

A company assistant that answers employees' questions from internal documents and never uses a
document the person asking cannot open. PostgreSQL row-level security enforces the permissions.
An evaluation suite measures answer quality and leaks under four enforcement methods, so the
question "can a model be told to respect permissions, or must the database enforce them" gets a
number.

## Results

Claude Haiku 5.5, 30 planted facts, each asked by an employee allowed to read its document and
by one who is not, 3 repeats. "Filter deleted" removes the permission filter from the
application code on purpose.

| Method | Filter deleted | Answered (of 30) | Fact reached the model (of 30) | Fact in the answer (of 30) |
|---|---|---|---|---|
| rls, row-level security in PostgreSQL | no | 30 | 0 | 0 |
| rls | yes | 30 | 0 | 0 |
| in_query, WHERE clause in the application's SQL | no | 30 | 0 | 0 |
| in_query | yes | 30 | 30 | 30 |
| post_filter, drop forbidden hits in Python | no | 30 | 0 | 0 |
| post_filter | yes | 30 | 30 | 30 |
| prompt_only, passages labelled with their readers, asker named in the prompt | no | 30 | 30 | 0 to 1 |

The full table with retrieval ranks and cost is in [results/results.md](results/results.md).
The run cost 0.39 USD.

## How it works

1. A Keycloak access token is validated. The username and groups become principals such as
   `user:nadia` and `group:hr`.
2. Each request opens a database transaction and sets `app.principals` for that transaction only.
3. Row-level policies on documents, chunks and access rows return a row only when one of its
   readers is in `app.principals`. The application role has SELECT only and cannot bypass the
   policies.
4. A LangGraph agent calls a search tool that takes a query and nothing else. The principals are
   set outside the model's reach.
5. The answer cites document ids. When nothing permitted is found, the reply is the fixed
   sentence "I found nothing you have access to on this."

Embeddings run locally with bge-small-en-v1.5 through fastembed. The stack is FastAPI,
PostgreSQL 17 with pgvector, Keycloak and Docker Compose.

## Run it

```
docker compose up -d --wait db keycloak
pip install -e .
python -m needtoknow.ingest
export ANTHROPIC_API_KEY=...
python -m needtoknow.agent --user nadia "What is the 2026 salary band for an HR manager?"
uvicorn --factory needtoknow.api:create_default_app
```

The corpus is a fictional company with 12 employees and about 40 documents. Every user's
password is `needtoknow` in the development realm.

## Evaluate

```
python -m needtoknow.evaluate
```

Writes `results/results.md`, `results/summary.json` and `results/raw.jsonl`. The run stops
when spending reaches `NEEDTOKNOW_BUDGET_USD`, 2 USD by default. CI runs the free checks
(ruff, mypy, 211 tests, including retrieval-level leak checks) on every pull request; the paid
run is on demand.

## Limits

- The corpus is fictional and small, with one planted fact per restricted document.
- One model and three repeats. No scale or latency numbers.
- The denied employee is a colleague without access, not an attacker with a crafted prompt.
- Some refusals carry extra commentary; the "existence disclosed" column in the results counts
  them.

## Status

Version 0.1 is in progress. Built and under review: tools that open a whole document and look
up colleagues under the same policies, an MCP server that searches as the signed-in employee,
and a chat page.
