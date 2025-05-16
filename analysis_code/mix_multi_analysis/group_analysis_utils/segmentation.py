import numpy as np
from constants import PARAMS

def segment_active_phases(death_lbl):
    """
    death_lbl : {prey_idx : 1/0 array}
    Returns list of (start,end) indices where ≥ min_prey_alive are alive.
    """
    prey=list(death_lbl)
    T=len(death_lbl[prey[0]]) if prey else 0
    alive=sum(death_lbl[p] for p in prey)
    segs=[]
    in_seg=False
    dead_run=0
    start = 0
    for t,a in enumerate(alive):
        if a >= PARAMS.min_prey_alive:
            dead_run=0
            if not in_seg:
              start=t
              in_seg=True
        else:
            if in_seg:
                dead_run = dead_run+1 if a==0 else 0
                if dead_run>=PARAMS.min_all_dead_duration:
                    segs.append((start, t-PARAMS.min_all_dead_duration+1))
                    in_seg=False
    if in_seg: segs.append((start,T))
    return segs
