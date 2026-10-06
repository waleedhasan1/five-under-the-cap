-- One row per player: identity and body measures from their latest season.

CREATE OR REPLACE TABLE dim_player AS
SELECT
    player_id,
    arg_max(player_name, season) AS player_name,
    arg_max(team_id, season) AS last_team_id,
    arg_max(team_abbreviation, season) AS last_team_abbreviation,
    min(season) AS first_season,
    max(season) AS last_season,
    arg_max(height_in, season) AS height_in,
    arg_max(weight_lb, season) AS weight_lb
FROM fct_player_season
GROUP BY player_id;
