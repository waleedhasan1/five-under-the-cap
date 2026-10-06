-- One row per five-man lineup per team-season. Lineups with no possessions are dropped.
-- net_rating is per 100 possessions; possessions are totals for the season.

CREATE OR REPLACE TABLE fct_lineup_season AS
WITH parsed AS (
    SELECT
        *,
        list_sort(
            list_transform(string_split(trim(group_id, '-'), '-'), x -> CAST(x AS BIGINT))
        ) AS ids
    FROM stg_lineups
)

SELECT
    season,
    team_id,
    team_abbreviation,
    ids[1] AS p1,
    ids[2] AS p2,
    ids[3] AS p3,
    ids[4] AS p4,
    ids[5] AS p5,
    gp,
    min AS minutes,
    poss AS possessions,
    net_rating,
    off_rating,
    def_rating
FROM parsed
WHERE poss > 0;
