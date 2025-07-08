def is_action_node(node):
    return "player" in node.agent


def policy_update(info_set):
    # Placeholder – depends on how policy is stored/used
    # For now this does nothing, as policy may be implicitly derived from regrets
    pass


def tree_traversal(node):
    # Recursively traverse and print or collect all nodes (optional)
    for child in node.children.values():
        tree_traversal(child)


def discountRegret(info_set, rate):
    for a in info_set:
        info_set[a] *= rate  # apply decay to each regret


def regret_matching(regret_map):
    regrets = [max(r, 0) for r in regret_map.values()]
    total = sum(regrets)
    if total > 0:
        return {a: max(regret_map[a], 0) / total for a in regret_map}
    else:
        num_actions = len(regret_map)
        return {a: 1 / num_actions for a in regret_map}


def get_infoset_key(node):
    # Create a hashable representation of the information set
    obs = tuple(node.observation['observation'])  # assuming array-like
    return (node.agent, obs)
