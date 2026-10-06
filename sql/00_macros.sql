-- Shared helpers for the mart transformations.

-- Name key used to join NBA.com and Basketball Reference players:
-- accents stripped, lower case, letters only, generational suffixes removed.
CREATE OR REPLACE MACRO norm_name(n) AS
    regexp_replace(
        trim(regexp_replace(regexp_replace(lower(strip_accents(n)), '[^a-z ]', '', 'g'), '\s+', ' ', 'g')),
        ' (jr|sr|ii|iii|iv|v)$', ''
    );

-- Basketball Reference team codes that differ from NBA.com's.
CREATE OR REPLACE MACRO nba_team_abbr(t) AS
    CASE t WHEN 'BRK' THEN 'BKN' WHEN 'CHO' THEN 'CHA' WHEN 'PHO' THEN 'PHX' ELSE t END;
