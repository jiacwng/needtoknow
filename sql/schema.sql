-- Tables, grants and row-level security policies. Run by needtoknow_owner; running it again
-- drops the tables and starts empty.

DROP VIEW IF EXISTS spending_by_month, forecast_vs_actual, open_tasks, my_meetings;
DROP TABLE IF EXISTS chunks, doc_access, documents, spending, forecasts, meeting_attendees,
    meetings, tasks, employees;
DROP FUNCTION IF EXISTS my_meeting_ids(), busy_times(TEXT[], TIMESTAMPTZ, TIMESTAMPTZ);

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

CREATE TABLE spending (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    department TEXT NOT NULL,
    vendor     TEXT NOT NULL,
    month      DATE NOT NULL CHECK (extract(day FROM month) = 1),
    amount_eur NUMERIC(12, 2) NOT NULL
);

CREATE TABLE forecasts (
    department TEXT NOT NULL,
    month      DATE NOT NULL CHECK (extract(day FROM month) = 1),
    amount_eur NUMERIC(12, 2) NOT NULL,
    PRIMARY KEY (department, month)
);

CREATE TABLE meetings (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organizer  TEXT NOT NULL REFERENCES employees (id),
    title      TEXT NOT NULL,
    starts_at  TIMESTAMPTZ NOT NULL,
    ends_at    TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (ends_at > starts_at)
);

CREATE TABLE meeting_attendees (
    meeting_id BIGINT NOT NULL REFERENCES meetings (id),
    employee   TEXT NOT NULL REFERENCES employees (id),
    PRIMARY KEY (meeting_id, employee)
);

CREATE INDEX meeting_attendees_employee ON meeting_attendees (employee, meeting_id);

CREATE TABLE tasks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    owner       TEXT NOT NULL REFERENCES employees (id),
    assignee    TEXT NOT NULL REFERENCES employees (id),
    due_date    DATE NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- These two functions run as the owner, so row-level security does not filter what they read.
-- That is a deliberate, narrow exception. my_meeting_ids returns only the caller's own meetings;
-- the meetings and attendee policies need it because each would otherwise read the other's
-- table and recurse. busy_times lets anyone schedule around a colleague: it returns when they
-- are busy and never a title or an attendee.
CREATE FUNCTION my_meeting_ids() RETURNS SETOF BIGINT
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public
AS $$
    SELECT id FROM meetings WHERE organizer = current_setting('app.user_id', true)
    UNION
    SELECT meeting_id FROM meeting_attendees WHERE employee = current_setting('app.user_id', true)
$$;

CREATE FUNCTION busy_times(employee_ids TEXT[], from_ts TIMESTAMPTZ, to_ts TIMESTAMPTZ)
RETURNS TABLE (employee TEXT, starts_at TIMESTAMPTZ, ends_at TIMESTAMPTZ)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public
AS $$
    SELECT busy.employee, meetings.starts_at, meetings.ends_at
    FROM meetings
    JOIN (
        SELECT id AS meeting_id, organizer AS employee FROM meetings
        UNION
        SELECT meeting_id, employee FROM meeting_attendees
    ) AS busy ON busy.meeting_id = meetings.id
    WHERE busy.employee = ANY (employee_ids)
      AND meetings.starts_at < to_ts
      AND meetings.ends_at > from_ts
      AND current_setting('app.user_id', true) <> ''
    ORDER BY busy.employee, meetings.starts_at
$$;

REVOKE EXECUTE ON FUNCTION my_meeting_ids(), busy_times(TEXT[], TIMESTAMPTZ, TIMESTAMPTZ)
FROM PUBLIC;
GRANT EXECUTE ON FUNCTION my_meeting_ids(), busy_times(TEXT[], TIMESTAMPTZ, TIMESTAMPTZ)
TO needtoknow_app;

CREATE VIEW spending_by_month WITH (security_invoker = true) AS
SELECT department, vendor, month, sum(amount_eur) AS amount_eur
FROM spending
GROUP BY department, vendor, month;

CREATE VIEW forecast_vs_actual WITH (security_invoker = true) AS
SELECT
    forecasts.department,
    forecasts.month,
    forecasts.amount_eur AS forecast_eur,
    actual.amount_eur AS actual_eur,
    actual.amount_eur - forecasts.amount_eur AS difference_eur
FROM forecasts
LEFT JOIN (
    SELECT department, month, sum(amount_eur) AS amount_eur
    FROM spending
    GROUP BY department, month
) AS actual USING (department, month);

CREATE VIEW open_tasks WITH (security_invoker = true) AS
SELECT id, title, owner, assignee, due_date
FROM tasks
WHERE status = 'open';

CREATE VIEW my_meetings WITH (security_invoker = true) AS
SELECT
    meetings.id,
    meetings.title,
    meetings.starts_at,
    meetings.ends_at,
    meetings.organizer,
    ARRAY(
        SELECT meeting_attendees.employee
        FROM meeting_attendees
        WHERE meeting_attendees.meeting_id = meetings.id
        ORDER BY meeting_attendees.employee
    ) AS attendees
FROM meetings;

GRANT SELECT ON documents, doc_access, chunks, employees, spending, forecasts, meetings,
    meeting_attendees, tasks TO needtoknow_app;
GRANT SELECT ON spending_by_month, forecast_vs_actual, open_tasks, my_meetings TO needtoknow_app;
GRANT INSERT ON meetings, meeting_attendees, tasks TO needtoknow_app;
GRANT UPDATE (status) ON tasks TO needtoknow_app;
GRANT SELECT ON documents, doc_access, chunks, employees, spending, forecasts, meetings,
    meeting_attendees, tasks TO needtoknow_reader;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE spending ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE meetings ENABLE ROW LEVEL SECURITY;
ALTER TABLE meeting_attendees ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;

-- app.principals and app.user_id are set per transaction by db.as_user. When they are unset the
-- list is NULL or empty and the user is NULL or empty, so no row matches.
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

CREATE POLICY spending_read ON spending FOR SELECT TO needtoknow_app
USING (
    string_to_array(current_setting('app.principals', true), ',')
        && ARRAY['group:finance', 'group:executives']
    OR ('group:' || department) = ANY (string_to_array(current_setting('app.principals', true), ','))
);

CREATE POLICY forecasts_read ON forecasts FOR SELECT TO needtoknow_app
USING (
    string_to_array(current_setting('app.principals', true), ',')
        && ARRAY['group:finance', 'group:executives']
    OR ('group:' || department) = ANY (string_to_array(current_setting('app.principals', true), ','))
);

CREATE POLICY meetings_read ON meetings FOR SELECT TO needtoknow_app
USING (
    organizer = current_setting('app.user_id', true)
    OR id IN (SELECT my_meeting_ids())
);

CREATE POLICY meetings_insert ON meetings FOR INSERT TO needtoknow_app
WITH CHECK (organizer = current_setting('app.user_id', true));

CREATE POLICY meeting_attendees_read ON meeting_attendees FOR SELECT TO needtoknow_app
USING (meeting_id IN (SELECT my_meeting_ids()));

CREATE POLICY meeting_attendees_insert ON meeting_attendees FOR INSERT TO needtoknow_app
WITH CHECK (
    EXISTS (
        SELECT 1
        FROM meetings
        WHERE meetings.id = meeting_attendees.meeting_id
          AND meetings.organizer = current_setting('app.user_id', true)
    )
);

CREATE POLICY tasks_read ON tasks FOR SELECT TO needtoknow_app
USING (
    owner = current_setting('app.user_id', true)
    OR assignee = current_setting('app.user_id', true)
    OR EXISTS (
        SELECT 1
        FROM employees
        WHERE employees.id = tasks.assignee
          AND employees.manager = current_setting('app.user_id', true)
    )
);

CREATE POLICY tasks_insert ON tasks FOR INSERT TO needtoknow_app
WITH CHECK (owner = current_setting('app.user_id', true) AND status = 'open');

CREATE POLICY tasks_update ON tasks FOR UPDATE TO needtoknow_app
USING (
    owner = current_setting('app.user_id', true) OR assignee = current_setting('app.user_id', true)
)
WITH CHECK (
    owner = current_setting('app.user_id', true) OR assignee = current_setting('app.user_id', true)
);

-- needtoknow_reader is the evaluation's model of an application the database does not protect:
-- it reads every row, and its own SQL or code must drop what the asker may not see.
CREATE POLICY documents_reader ON documents FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY doc_access_reader ON doc_access FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY chunks_reader ON chunks FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY employees_reader ON employees FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY spending_reader ON spending FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY forecasts_reader ON forecasts FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY meetings_reader ON meetings FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY meeting_attendees_reader ON meeting_attendees FOR SELECT TO needtoknow_reader
USING (true);
CREATE POLICY tasks_reader ON tasks FOR SELECT TO needtoknow_reader USING (true);
