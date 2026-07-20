"""Adapter from the upstream SMACv2 API to DG-MAPPO's SMAC interface."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import yaml
from gym.spaces import Discrete


_CONFIG_DIR = (
    Path(__file__).resolve().parent
    / "smacv2"
    / "smacv2"
    / "examples"
    / "configs"
)


def resolve_smacv2_config_path(config_name):
    """Resolve a bundled config alias, filename, or explicit YAML path."""
    requested = Path(config_name).expanduser()
    if requested.is_file():
        return requested.resolve()

    name = requested.name
    aliases = {
        "terran": "sc2_gen_terran.yaml",
        "protoss": "sc2_gen_protoss.yaml",
        "zerg": "sc2_gen_zerg.yaml",
        "terran_epo": "sc2_gen_terran_epo.yaml",
        "protoss_epo": "sc2_gen_protoss_epo.yaml",
        "zerg_epo": "sc2_gen_zerg_epo.yaml",
    }
    name = aliases.get(name, name)
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"

    candidate = _CONFIG_DIR / name
    if candidate.is_file():
        return candidate

    available = ", ".join(sorted(aliases))
    raise FileNotFoundError(
        f"SMACv2 config '{config_name}' was not found. Use a YAML path or one "
        f"of: {available}."
    )


def load_smacv2_env_args(
    config_name="terran_epo",
    n_units=None,
    n_enemies=None,
    prob_obs_enemy=None,
    action_mask=None,
):
    """Load SMACv2 ``env_args`` and apply command-line overrides."""
    config_path = resolve_smacv2_config_path(config_name)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("env_args"), dict):
        raise ValueError(f"{config_path} does not contain an 'env_args' mapping")

    env_args = deepcopy(config["env_args"])
    capability_config = env_args.get("capability_config")
    if not isinstance(capability_config, dict):
        raise ValueError(f"{config_path} does not define a capability_config")

    if n_units is not None:
        if n_units <= 0:
            raise ValueError("--smacv2_n_units must be greater than zero")
        capability_config["n_units"] = int(n_units)
    if n_enemies is not None:
        if n_enemies <= 0:
            raise ValueError("--smacv2_n_enemies must be greater than zero")
        capability_config["n_enemies"] = int(n_enemies)
    if prob_obs_enemy is not None:
        if not 0.0 <= prob_obs_enemy <= 1.0:
            raise ValueError("--smacv2_prob_obs_enemy must be in [0, 1]")
        env_args["prob_obs_enemy"] = float(prob_obs_enemy)
    if action_mask is not None:
        env_args["action_mask"] = bool(action_mask)

    race = str(env_args.get("map_name", "smacv2")).split("_")[-1]
    n_allies = capability_config["n_units"]
    n_opponents = capability_config["n_enemies"]
    is_epo = (
        float(env_args.get("prob_obs_enemy", 1.0)) < 1.0
        or not bool(env_args.get("action_mask", True))
    )
    scenario_name = f"{race}_{n_allies}_vs_{n_opponents}"
    if is_epo:
        scenario_name += "_epo"

    return env_args, scenario_name, config_path


def _create_upstream_env(env_args):
    try:
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    except ImportError as error:
        raise ImportError(
            "The upstream SMACv2 package is required. Install it with "
            "`pip install -r requirements-smacv2.txt`."
        ) from error

    # SMACv2's wrapper mutates nested distribution configs during construction.
    return StarCraftCapabilityEnvWrapper(**deepcopy(env_args))


class SMACv2EnvAdapter:
    """Expose SMACv2 through the six-value interface used by this repository.

    The official environment provides one global state and one team reward. This
    adapter repeats them per agent, as expected by ``ShareSubprocVecEnv``. For
    graph algorithms it builds an undirected ally graph from physical sight
    ranges. EPO enemy masking remains entirely owned by upstream SMACv2.
    """

    def __init__(self, args, env_args=None, seed=None, env=None):
        if env is None:
            if env_args is None:
                raise ValueError(
                    "env_args are required when no SMACv2 env is injected"
                )
            upstream_args = deepcopy(env_args)
            if seed is not None:
                upstream_args["seed"] = int(seed)
            env = _create_upstream_env(upstream_args)

        self.env = env
        self._seed = seed
        self._stack_frames = bool(getattr(args, "use_stacked_frames", False))
        self._n_frames = int(getattr(args, "stacked_frames", 1))
        if self._n_frames <= 0:
            raise ValueError("stacked_frames must be greater than zero")

        comm_range = getattr(args, "smacv2_comm_range", None)
        self.communication_range = None if comm_range is None else float(comm_range)
        if self.communication_range is not None and self.communication_range <= 0:
            raise ValueError("--smacv2_comm_range must be greater than zero")
        self.force_connected_graph = bool(
            getattr(args, "smacv2_force_connected_graph", False)
        )

        env_info = self.env.get_env_info()
        self.n_agents = int(env_info["n_agents"])
        self.n_enemies = int(
            getattr(
                self.env,
                "n_enemies",
                env_args.get("capability_config", {}).get("n_enemies", 0)
                if env_args
                else 0,
            )
        )
        self.n_actions = int(env_info["n_actions"])
        self.episode_limit = int(env_info["episode_limit"])
        self._obs_size = int(env_info["obs_shape"])
        self._state_size = int(env_info["state_shape"])

        frame_multiplier = self._n_frames if self._stack_frames else 1
        obs_space = [self._obs_size * frame_multiplier]
        state_space = [self._state_size * frame_multiplier]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]
        self.observation_space = [list(obs_space) for _ in range(self.n_agents)]
        self.share_observation_space = [
            list(state_space) for _ in range(self.n_agents)
        ]

        self._obs_frames = None
        self._state_frames = None
        self.seed(seed)

    @property
    def unwrapped(self):
        return self

    def _seed_distributions(self, seed):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        distribution_map = getattr(self.env, "env_key_to_distribution_map", {})
        visited = set()
        next_seed = [seed + 1]

        def seed_object(value):
            object_id = id(value)
            if object_id in visited or not hasattr(value, "__dict__"):
                return
            visited.add(object_id)
            for attribute, child in vars(value).items():
                if isinstance(child, np.random.Generator):
                    setattr(value, attribute, np.random.default_rng(next_seed[0]))
                    next_seed[0] += 1
                elif child.__class__.__module__.startswith("smacv2."):
                    seed_object(child)

        for distribution in distribution_map.values():
            seed_object(distribution)

    def seed(self, seed):
        self._seed = None if seed is None else int(seed)
        self._seed_distributions(self._seed)
        target = getattr(self.env, "env", self.env)
        if self._seed is not None and hasattr(target, "_seed"):
            target._seed = self._seed
        return self._seed

    def _format_obs_and_state(self, observations, state, reset=False):
        observations = np.asarray(observations, dtype=np.float32).reshape(
            self.n_agents, self._obs_size
        )
        state = np.asarray(state, dtype=np.float32).reshape(self._state_size)
        shared_state = np.repeat(state[None, :], self.n_agents, axis=0)

        if not self._stack_frames:
            return observations, shared_state

        if reset or self._obs_frames is None:
            self._obs_frames = np.zeros(
                (self.n_agents, self._n_frames, self._obs_size), dtype=np.float32
            )
            self._state_frames = np.zeros(
                (self.n_agents, self._n_frames, self._state_size), dtype=np.float32
            )
        else:
            self._obs_frames = np.roll(self._obs_frames, -1, axis=1)
            self._state_frames = np.roll(self._state_frames, -1, axis=1)

        self._obs_frames[:, -1] = observations
        self._state_frames[:, -1] = shared_state
        return (
            self._obs_frames.reshape(self.n_agents, -1),
            self._state_frames.reshape(self.n_agents, -1),
        )

    def _available_actions(self):
        return np.asarray(self.env.get_avail_actions(), dtype=np.float32)

    def reset(self):
        observations, state = self.env.reset()
        observations, shared_state = self._format_obs_and_state(
            observations, state, reset=True
        )
        return observations, shared_state, self._available_actions()

    def _agent_is_alive(self, agent_id):
        try:
            return self.env.get_unit_by_id(agent_id).health > 0
        except (AttributeError, KeyError):
            agents = getattr(self.env, "agents", {})
            return agent_id in agents and agents[agent_id].health > 0

    def _normalise_info(self, raw_info, terminated):
        raw_info = dict(raw_info or {})
        # Upstream SMACv2 1.0.0 divides battles_won by battles_game in
        # get_stats(), which raises before the first episode has completed.
        # The counters we need are public environment attributes, so reading
        # them directly is both cheaper and safe when battles_game is zero.
        battles_game = getattr(self.env, "battles_game", None)
        if battles_game is not None:
            stats = {
                "battles_won": getattr(self.env, "battles_won", 0),
                "battles_game": battles_game,
                "timeouts": getattr(self.env, "timeouts", 0),
                "restarts": getattr(self.env, "force_restarts", 0),
            }
        else:
            try:
                stats = dict(self.env.get_stats())
            except (AttributeError, ArithmeticError, TypeError):
                stats = {}

        won = bool(raw_info.get("battle_won", raw_info.get("won", False)))
        common = dict(raw_info)
        common.update(
            battles_won=stats.get("battles_won", int(won)),
            battles_game=stats.get("battles_game", int(bool(terminated))),
            battles_draw=stats.get("timeouts", stats.get("battles_draw", 0)),
            restarts=stats.get("restarts", 0),
            bad_transition=bool(raw_info.get("episode_limit", False)),
            won=won,
        )
        return [dict(common) for _ in range(self.n_agents)]

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.n_agents).astype(np.int64).tolist()
        reward, terminated, raw_info = self.env.step(actions)
        observations = self.env.get_obs()
        state = self.env.get_state()
        observations, shared_state = self._format_obs_and_state(observations, state)

        dones = np.asarray(
            [bool(terminated) or not self._agent_is_alive(i) for i in range(self.n_agents)],
            dtype=np.bool_,
        )
        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)
        infos = self._normalise_info(raw_info, terminated)
        return (
            observations,
            shared_state,
            rewards,
            dones,
            infos,
            self._available_actions(),
        )

    @staticmethod
    def _distance(first, second):
        return float(
            np.hypot(first.pos.x - second.pos.x, first.pos.y - second.pos.y)
        )

    def _alive_agent_ids(self):
        return [
            agent_id
            for agent_id in range(self.n_agents)
            if self._agent_is_alive(agent_id)
        ]

    def _pair_communication_range(self, first_id, second_id):
        if self.communication_range is not None:
            return self.communication_range
        return min(
            float(self.env.unit_sight_range(first_id)),
            float(self.env.unit_sight_range(second_id)),
        )

    def get_agent_communication_matrix(self):
        adjacency = np.zeros((self.n_agents, self.n_agents), dtype=np.bool_)
        alive_ids = self._alive_agent_ids()
        agents = getattr(self.env, "agents", {})
        for offset, first_id in enumerate(alive_ids):
            for second_id in alive_ids[offset + 1 :]:
                distance = self._distance(agents[first_id], agents[second_id])
                if distance < self._pair_communication_range(first_id, second_id):
                    adjacency[first_id, second_id] = True
                    adjacency[second_id, first_id] = True

        if self.force_connected_graph:
            self._connect_components(adjacency, alive_ids, agents)
        return adjacency

    @staticmethod
    def _components(adjacency, node_ids):
        unseen = set(node_ids)
        components = []
        while unseen:
            root = min(unseen)
            unseen.remove(root)
            component = [root]
            stack = [root]
            while stack:
                node = stack.pop()
                neighbors = sorted(
                    neighbor
                    for neighbor in tuple(unseen)
                    if adjacency[node, neighbor]
                )
                for neighbor in neighbors:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
            components.append(component)
        return components

    def _connect_components(self, adjacency, alive_ids, agents):
        components = self._components(adjacency, alive_ids)
        if len(components) <= 1:
            return
        connected = components.pop(0)
        while components:
            candidates = []
            for component_index, component in enumerate(components):
                for source in connected:
                    for target in component:
                        candidates.append(
                            (
                                self._distance(agents[source], agents[target]),
                                source,
                                target,
                                component_index,
                            )
                        )
            _, source, target, component_index = min(candidates)
            adjacency[source, target] = adjacency[target, source] = True
            connected.extend(components.pop(component_index))

    def get_visibility_matrix(self):
        # Only the ally columns are consumed by the graph algorithms. EPO enemy
        # visibility is deliberately not reconstructed here, which avoids
        # leaking SMACv2's persistent stochastic enemy masks.
        visibility = np.zeros(
            (self.n_agents, self.n_agents + self.n_enemies), dtype=np.bool_
        )
        visibility[:, : self.n_agents] = self.get_agent_communication_matrix()
        return visibility

    def get_edge_index_matrix(self, faulty_node=None):
        del faulty_node
        adjacency = self.get_agent_communication_matrix()
        edge_index = np.zeros((2, self.n_agents * self.n_agents), dtype=np.int64)
        edge_index[1, :] = -1
        for agent_id in range(self.n_agents):
            start = agent_id * self.n_agents
            neighbors = np.flatnonzero(adjacency[agent_id])
            edge_index[0, start : start + self.n_agents] = agent_id
            edge_index[1, start] = agent_id
            edge_index[1, start + 1 : start + 1 + len(neighbors)] = neighbors
        return edge_index

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        return self.env.save_replay()

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
