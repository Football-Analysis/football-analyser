from dataclasses import dataclass


@dataclass
class Prediction:
    match_id: str
    home: float
    away: float
    draw: float