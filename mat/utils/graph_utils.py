import numpy as np


def assert_connected_ally_topologies(adjacencies, alive_masks):
    """Verify one connected undirected component over every alive ally set."""
    adjacencies = np.asarray(adjacencies, dtype=np.bool_)
    alive_masks = np.asarray(alive_masks, dtype=np.bool_)
    if adjacencies.ndim != 3 or adjacencies.shape[1] != adjacencies.shape[2]:
        raise RuntimeError(
            f"Expected batched square ally adjacencies, got {adjacencies.shape}."
        )
    if alive_masks.shape != adjacencies.shape[:2]:
        raise RuntimeError(
            "Alive-agent mask does not match ally adjacency batch: "
            f"{alive_masks.shape} != {adjacencies.shape[:2]}."
        )

    for env_id, (adjacency, alive_mask) in enumerate(
        zip(adjacencies, alive_masks)
    ):
        alive_ids = np.flatnonzero(alive_mask)
        if alive_ids.size <= 1:
            continue
        alive_adjacency = adjacency[np.ix_(alive_ids, alive_ids)]
        if not np.array_equal(alive_adjacency, alive_adjacency.T):
            raise RuntimeError(
                f"Ally communication graph for environment {env_id} is directed."
            )

        visited = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(alive_adjacency[node]):
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(visited) != alive_ids.size:
            unreachable = alive_ids[
                [index for index in range(alive_ids.size) if index not in visited]
            ].tolist()
            raise RuntimeError(
                "Disconnected ally communication graph for environment "
                f"{env_id}; unreachable alive agents: {unreachable}."
            )
