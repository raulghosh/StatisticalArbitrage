"""
src/signals — Signal generation layer.

Public API
----------
    from src.signals import SignalEvent, SignalAction, PositionState
    from src.signals import ZScoreSignal
    from src.signals import OUOptimalSignal, OUBoundaries

Modules
-------
base.py       : SignalEvent, PositionState, SignalAction, BaseSignal
zscore.py     : ZScoreSignal    — fixed z-score thresholds + state machine
ou_optimal.py : OUOptimalSignal — Bertram/numerical optimal O-U boundaries

Planned
-------
momentum.py   : Spread momentum / breakout signal
ensemble.py   : Multi-signal combination with conviction weighting
"""

from src.signals.base import (
    BaseSignal,
    PositionState,
    SignalAction,
    SignalEvent,
)
from src.signals.zscore import ZScoreSignal
from src.signals.ou_optimal import OUOptimalSignal, OUBoundaries

__all__ = [
    # Base contracts
    "BaseSignal",
    "PositionState",
    "SignalAction",
    "SignalEvent",
    # Signals
    "ZScoreSignal",
    "OUOptimalSignal",
    "OUBoundaries",
]
