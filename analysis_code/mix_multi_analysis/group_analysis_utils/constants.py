"""
Centralised parameters & stamina tables.
Edit here – no magic numbers elsewhere.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    # segmentation
    min_prey_alive: int = 1
    min_all_dead_duration: int = 5

    # collective apple (point)
    apple_radius: int = 3
    apple_min_prey: int = 2
    apple_cooldown: int = 5

    # collective apple (period)
    apple_period_radius: int = 3
    apple_period_min_len: int = 5
    apple_period_gap: int = 2

    # distraction
    lurk_radius: int = 3
    shift_window: int = 5
    dist_switch: int = 4
    moveaway_win: int = 5
    moveaway_majority: int = 3
    distraction_cap: int = 30

PARAMS = Params()    # import this everywhere
