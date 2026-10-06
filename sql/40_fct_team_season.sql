-- One row per team-season: record and per-100 ratings.

CREATE OR REPLACE TABLE fct_team_season AS
SELECT
    b.season,
    b.team_id,
    b.team_name,
    b.gp,
    b.w AS wins,
    b.l AS losses,
    a.net_rating,
    a.off_rating,
    a.def_rating,
    a.pace,
    a.poss AS possessions
FROM stg_team_base AS b
JOIN stg_team_advanced AS a USING (season, team_id);
