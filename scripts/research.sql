-- duckdb -init scripts/research.sql   (run from the repo root)
-- Receipts are parameters, games are facts, metrics are these queries.
CREATE OR REPLACE VIEW runs AS
  SELECT * FROM read_json_auto('runs/*/spec.json', union_by_name=true);
CREATE OR REPLACE VIEW events AS
  SELECT * FROM read_json_auto('runs/*/events.jsonl', union_by_name=true);
CREATE OR REPLACE VIEW games   AS SELECT * FROM events WHERE kind = 'game';
CREATE OR REPLACE VIEW results AS SELECT * FROM events WHERE kind = 'result';

-- Examples (the doctrine demo — headline metrics as queries over raw games):
--   death rate by bot, all versus games ever:
--     SELECT a, avg(a_topped::INT) FROM games WHERE mode='versus' GROUP BY a;
--   recompute the censored downstack mean from facts + parameters:
--     SELECT g.run, avg(CASE WHEN g.cleared THEN g.pieces ELSE r.spec.max_pieces END)
--     FROM games g JOIN runs r ON g.run = r.run_id WHERE g.mode='downstack' GROUP BY g.run;
--   exclude dirty-tree runs from anything:
--     ... JOIN runs r ON g.run = r.run_id WHERE NOT r.git.dirty ...
