#!/usr/bin/env bash
set -euo pipefail

export SC2PATH="${SC2PATH:-/opt/StarCraftII}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

sc2_binary=$(find "${SC2PATH}/Versions" -maxdepth 2 -type f -name SC2_x64 \
  -perm -u+x -print -quit 2>/dev/null)
test -n "${sc2_binary}"
test -f "${SC2PATH}/Maps/SMAC_Maps/32x32_flat.SC2Map"

python - <<'PY'
import numpy as np
import torch
import torch_geometric
import torch_scatter
import smacv2
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

assert torch.cuda.is_available(), "PyTorch cannot access the assigned GPU"
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyG: {torch_geometric.__version__}")
print(f"SMACv2: {smacv2.__file__}")

capability_config = {
    "n_units": 5,
    "n_enemies": 5,
    "team_gen": {
        "dist_type": "weighted_teams",
        "unit_types": ["marine", "marauder", "medivac"],
        "weights": [0.45, 0.45, 0.1],
        "exception_unit_types": ["medivac"],
        "observe": True,
    },
    "start_positions": {
        "dist_type": "surrounded_and_reflect",
        "p": 0.5,
        "map_x": 32,
        "map_y": 32,
    },
}

env = StarCraftCapabilityEnvWrapper(
    capability_config=capability_config,
    map_name="10gen_terran",
    debug=False,
    conic_fov=False,
    use_unit_ranges=True,
    min_attack_range=2,
    obs_own_pos=True,
    fully_observable=False,
)

try:
    env.reset()
    info = env.get_env_info()
    actions = []
    for agent_id in range(info["n_agents"]):
        available = np.flatnonzero(env.get_avail_agent_actions(agent_id))
        actions.append(int(available[0]))
    reward, terminated, step_info = env.step(actions)
    print(f"SMACv2 reset/step passed: agents={info['n_agents']}, reward={reward}, terminated={terminated}")
finally:
    env.close()

print("CONTAINER SMOKE TEST PASSED")
PY
