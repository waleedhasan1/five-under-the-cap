-- Salaries in long form, one row per Basketball Reference player and season, matched to
-- NBA.com player IDs by normalized name with team as the tie-breaker. Manual overrides win.

-- Long form: one row per source row and season with a non-empty salary.
CREATE OR REPLACE TABLE int_salary_long AS
SELECT
    bbref_id,
    player AS player_name,
    nba_team_abbr(team) AS team,
    replace(season, '_', '-') AS season,
    CAST(regexp_replace(salary_text, '[^0-9]', '', 'g') AS BIGINT) AS salary
FROM (
    UNPIVOT stg_salary
    ON COLUMNS('^\d{4}_\d{2}$')
    INTO NAME season VALUE salary_text
)
WHERE regexp_replace(salary_text, '[^0-9]', '', 'g') <> '';

-- Waived players appear once per team that pays them; their salary figure repeats.
-- Keep one row per player-season and record how many rows and distinct values there were.
CREATE OR REPLACE TABLE int_salary_dedup AS
SELECT
    bbref_id,
    season,
    any_value(player_name) AS player_name,
    list_sort(list_distinct(list(team))) AS teams,
    max(salary) AS salary,
    count(*) AS source_rows,
    count(DISTINCT salary) AS distinct_salaries
FROM int_salary_long
GROUP BY bbref_id, season;

CREATE OR REPLACE TABLE map_player_id AS
WITH bb AS (
    SELECT bbref_id, any_value(player_name) AS player_name,
        list_distinct(flatten(list(teams))) AS teams
    FROM int_salary_dedup
    GROUP BY bbref_id
),

candidates AS (
    SELECT
        bb.bbref_id,
        p.player_id,
        list_contains(bb.teams, p.last_team_abbreviation) AS team_match,
        count(*) OVER (PARTITION BY bb.bbref_id) AS n_candidates,
        row_number() OVER (
            PARTITION BY bb.bbref_id
            ORDER BY list_contains(bb.teams, p.last_team_abbreviation) DESC, p.last_season DESC
        ) AS rk
    FROM bb
    JOIN dim_player AS p ON norm_name(bb.player_name) = norm_name(p.player_name)
)

SELECT
    bb.bbref_id,
    bb.player_name,
    bb.teams,
    coalesce(o.nba_player_id, c.player_id) AS player_id,
    CASE
        WHEN o.nba_player_id IS NOT NULL THEN 'override'
        WHEN c.player_id IS NULL THEN 'unmatched'
        WHEN c.n_candidates = 1 THEN 'name'
        WHEN c.team_match THEN 'name+team'
        ELSE 'ambiguous'
    END AS match_method
FROM bb
LEFT JOIN candidates AS c ON c.bbref_id = bb.bbref_id AND c.rk = 1
LEFT JOIN stg_player_id_overrides AS o ON o.bbref_id = bb.bbref_id;

CREATE OR REPLACE TABLE dim_salary AS
SELECT
    m.player_id,
    s.bbref_id,
    s.player_name,
    s.season,
    s.salary,
    s.teams,
    s.source_rows
FROM int_salary_dedup AS s
JOIN map_player_id AS m USING (bbref_id);
