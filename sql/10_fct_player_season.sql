-- One row per player-season with the archetype features (PROJECT_PLAN.md 7.1).
-- All rate and share columns are fractions in [0, 1] except *_per100.

CREATE OR REPLACE TABLE fct_player_season AS
WITH zones AS (
    SELECT
        season,
        player_id,
        coalesce(restricted_area_fga, 0) AS rim,
        coalesce(in_the_paint_non_ra_fga, 0) AS paint,
        coalesce(mid_range_fga, 0) AS mid,
        coalesce(corner_3_fga, 0) AS c3,
        coalesce(above_the_break_3_fga, 0) AS atb3,
        coalesce(restricted_area_fga, 0) + coalesce(in_the_paint_non_ra_fga, 0)
            + coalesce(mid_range_fga, 0) + coalesce(corner_3_fga, 0)
            + coalesce(above_the_break_3_fga, 0) AS zone_fga
    FROM stg_player_shot_zones
),

-- Traded players have one Synergy row per team; sum possessions first.
play_poss AS (
    SELECT season, player_id, 'pt_' || lower(variant) AS play_type, sum(poss) AS poss
    FROM stg_player_play_types
    GROUP BY ALL
),

play_share AS (
    PIVOT (
        SELECT season, player_id, play_type,
            poss / sum(poss) OVER (PARTITION BY season, player_id) AS share
        FROM play_poss
    )
    ON play_type USING first(share)
)

SELECT
    b.season,
    b.player_id,
    b.player_name,
    b.team_id,
    b.team_abbreviation,
    b.age,
    b.gp,
    b.min AS minutes,
    a.poss AS possessions,
    bio.player_height_inches AS height_in,
    try_cast(bio.player_weight AS DOUBLE) AS weight_lb,
    z.rim / nullif(z.zone_fga, 0) AS zone_rim,
    z.paint / nullif(z.zone_fga, 0) AS zone_paint,
    z.mid / nullif(z.zone_fga, 0) AS zone_mid,
    z.c3 / nullif(z.zone_fga, 0) AS zone_corner3,
    z.atb3 / nullif(z.zone_fga, 0) AS zone_atb3,
    b.fg3a / nullif(b.fga, 0) AS fg3a_rate,
    b.fta / nullif(b.fga, 0) AS fta_rate,
    s.pct_ast_fgm AS assisted_share,
    a.usg_pct,
    a.ast_pct,
    a.oreb_pct,
    a.dreb_pct,
    100.0 * b.stl / nullif(a.poss, 0) AS stl_per100,
    100.0 * b.blk / nullif(a.poss, 0) AS blk_per100,
    p.* EXCLUDE (season, player_id)
FROM stg_player_base AS b
LEFT JOIN stg_player_advanced AS a USING (season, player_id)
LEFT JOIN stg_player_scoring AS s USING (season, player_id)
LEFT JOIN stg_player_bio AS bio USING (season, player_id)
LEFT JOIN zones AS z USING (season, player_id)
LEFT JOIN play_share AS p USING (season, player_id);
