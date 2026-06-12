-- duckdb -init scripts/research.sql   (run from the repo root)
-- Storage is normalized (receipts = parameters, games.jsonl = facts);
-- these views are the denormalization layer, reconstructed on read.
CREATE OR REPLACE VIEW runs AS
  SELECT regexp_extract(filename, 'runs/([^/]+)/', 1) AS run, *
  FROM read_json_auto('runs/*/spec.json', filename=true, union_by_name=true);
CREATE OR REPLACE VIEW games AS
  SELECT regexp_extract(filename, 'runs/([^/]+)/', 1) AS run, *
  FROM read_json_auto('runs/*/games.jsonl', filename=true, union_by_name=true);
-- Games with parameters joined back and versus bot names reconstructed
-- (receipt bots[1]/bots[2] + the per-game swapped bit).
CREATE OR REPLACE VIEW games_wide AS
  SELECT g.*, r.experiment, r.spec, r.git.dirty AS dirty,
         CASE WHEN g.swapped THEN r.bots[2] ELSE r.bots[1] END AS a,
         CASE WHEN g.swapped THEN r.bots[1] ELSE r.bots[2] END AS b,
         r.bots[1] AS solo_bot
  FROM games g JOIN runs r USING (run);

-- Examples (metrics are queries — never stored):
--   death rate by bot over every versus/race game ever:
--     SELECT a, avg(a_topped::INT) FROM games_wide WHERE a_topped IS NOT NULL GROUP BY a;
--   recompute a run's censored downstack mean from facts + parameters:
--     SELECT run, avg(CASE WHEN cleared THEN pieces ELSE spec.max_pieces END)
--     FROM games_wide WHERE cleared IS NOT NULL GROUP BY run;
--   a race's LLR trajectory: fold games ordered by n within its run.
