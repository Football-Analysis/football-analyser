from dataclasses import dataclass


@dataclass
class Match:
    date: str
    home_team: str
    away_team: str
    score: dict
    game_week: str
    season: int
    league: dict
