-- Tables, grants and row-level security policies. Run by needtoknow_owner; running it again
-- drops the tables and starts empty.

DROP TABLE IF EXISTS chunks, doc_access, documents, employees;

CREATE TABLE documents (
    id    TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body  TEXT NOT NULL
);

CREATE TABLE doc_access (
    doc_id    TEXT NOT NULL REFERENCES documents (id),
    principal TEXT NOT NULL,
    PRIMARY KEY (doc_id, principal)
);

CREATE INDEX doc_access_principal_doc_id ON doc_access (principal, doc_id);

CREATE TABLE chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents (id),
    position  INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE employees (
    id      TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    title   TEXT NOT NULL,
    groups  TEXT[] NOT NULL,
    manager TEXT REFERENCES employees (id) DEFERRABLE INITIALLY DEFERRED
);

GRANT SELECT ON documents, doc_access, chunks, employees TO needtoknow_app;
GRANT SELECT ON documents, doc_access, chunks, employees TO needtoknow_reader;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;

-- app.principals is set per transaction by db.as_user. When it is unset the list is NULL or
-- empty, so no row matches.
CREATE POLICY doc_access_read ON doc_access FOR SELECT TO needtoknow_app
USING (
    principal = ANY (string_to_array(current_setting('app.principals', true), ','))
);

CREATE POLICY documents_read ON documents FOR SELECT TO needtoknow_app
USING (
    EXISTS (
        SELECT 1
        FROM doc_access
        WHERE doc_access.doc_id = documents.id
          AND doc_access.principal = ANY (string_to_array(current_setting('app.principals', true), ','))
    )
);

CREATE POLICY chunks_read ON chunks FOR SELECT TO needtoknow_app
USING (
    EXISTS (
        SELECT 1
        FROM doc_access
        WHERE doc_access.doc_id = chunks.doc_id
          AND doc_access.principal = ANY (string_to_array(current_setting('app.principals', true), ','))
    )
);

CREATE POLICY employees_read ON employees FOR SELECT TO needtoknow_app
USING (
    'group:everyone' = ANY (string_to_array(current_setting('app.principals', true), ','))
);

-- needtoknow_reader is the evaluation's model of an application the database does not protect:
-- it reads every row, and its own SQL or code must drop what the asker may not see.
CREATE POLICY documents_reader ON documents FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY doc_access_reader ON doc_access FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY chunks_reader ON chunks FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY employees_reader ON employees FOR SELECT TO needtoknow_reader USING (true);
