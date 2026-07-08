"""Long-range cooperative predator-prey environments."""

from gymnasium.envs.registration import register

from mat.envs.long_range_predator_prey.continuous import (
    LongRangePredatorPreyContinuousEnv,
    LongRangePredatorPreyConfig,
    LongRangePredatorPreyTorchCore,
)


register(
    id="LongRangePredatorPreyContinuous-v0",
    entry_point="mat.envs.long_range_predator_prey.continuous:LongRangePredatorPreyContinuousEnv",
)


__all__ = [
    "LongRangePredatorPreyConfig",
    "LongRangePredatorPreyContinuousEnv",
    "LongRangePredatorPreyTorchCore",
]
