"""Salary parsing tests. The HTML below is a small hand-built FIXTURE, not real data."""

# ruff: noqa: E501

from fivecap.ingest.salaries import parse_contracts_html

FIXTURE = """
<table id="player-contracts"><thead>
<tr><th data-stat="ranker">Rk</th><th data-stat="player">Player</th><th data-stat="team_id">Tm</th>
<th aria-label="2026-27" data-stat="y1">2026-27</th><th aria-label="2027-28" data-stat="y2">2027-28</th>
<th data-stat="remain_gtd">Guaranteed</th></tr></thead><tbody>
<tr><th data-stat="ranker">1</th><td data-stat="player"><a href="/players/x/fixtu01.html">Fixture One</a></td>
<td data-stat="team_id">AAA</td><td data-stat="y1">$10,000,000</td><td data-stat="y2"></td>
<td data-stat="remain_gtd">$10,000,000</td></tr>
<tr class="thead"><th data-stat="ranker">Rk</th><td data-stat="player">Player</td></tr>
<tr><th data-stat="ranker">2</th><td data-stat="player"><a href="/players/x/fixtu02.html">Fixture Two</a></td>
<td data-stat="team_id">BBB</td><td data-stat="y1">$2,000,000</td><td data-stat="y2">$2,100,000</td>
<td data-stat="remain_gtd">$4,100,000</td></tr>
</tbody></table>
"""


def test_parse_contracts_html() -> None:
    df = parse_contracts_html(FIXTURE)
    assert list(df.columns) == ["player", "bbref_id", "team", "2026-27", "2027-28", "remain_gtd"]
    assert df["bbref_id"].tolist() == ["fixtu01", "fixtu02"]  # repeated header row dropped
    assert df.loc[1, "2027-28"] == "$2,100,000"
