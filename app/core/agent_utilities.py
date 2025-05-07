from copy import deepcopy

from app.utils.constants import DEFAULT_OPTIMISTIC_ALPHA


def marginal_contribution_utility(agent, new_action, coverage_map, M):
    """
    Agents receive value for a resource if the exact max_cover is achieved.
        If coverage > M, 0 reward.
        If coverage < M, 0 reward.
        If coverage == M, full reward.

    :param agent: current 'awake' agent
    :param new_action: candidate action to be scored
    :param coverage_map: current state of the system
    :param M: max_cover
    :return: difference between the agents current action, and the candidate action
    """
    current_action = agent.current_action
    previous_coverage = filter_coverage_map(coverage_map, current_action)

    old_ids = {r.id for r in current_action}
    new_ids = {r.id for r in new_action}

    new_action_score = 0
    prev_action_score = 0

    # score resources newly added
    for rid in new_ids - old_ids:
        cov = previous_coverage.get(rid, 0) + 1
        if cov == M:
            rv = next(r.value for r in new_action if r.id == rid)
            new_action_score += rv

    # subtract score for resources removed
    for rid in old_ids - new_ids:
        cov = coverage_map.get(rid, 0)
        if cov == M:
            rv = next(r.value for r in current_action if r.id == rid)
            prev_action_score += rv

    return new_action_score - prev_action_score


def equal_share_utility(agent, new_action, coverage_map, M):
    """
    Agents receive value for a resource if the exact max_cover is achieved or exceeded.
        If coverage >= M, 1/num_agents reward.
        If coverage < M, 0 reward.

    :param agent: current 'awake' agent
    :param new_action: candidate action to be scored
    :param coverage_map: current state of the system
    :param M: max_cover
    :return: difference between the agents current action, and the candidate action
    """
    current_action = agent.current_action
    previous_coverage = filter_coverage_map(coverage_map, current_action)

    old_ids = {r.id for r in current_action}
    new_ids = {r.id for r in new_action}

    new_action_score = 0
    prev_action_score = 0

    # score resources newly added
    for rid in new_ids - old_ids:
        cov = previous_coverage.get(rid, 0) + 1
        if cov >= M:
            rv = next(r.value for r in new_action if r.id == rid)
            new_action_score += rv / cov

    # subtract score for resources removed
    for rid in old_ids - new_ids:
        cov = coverage_map.get(rid, 0)
        if cov >= M:
            rv = next(r.value for r in current_action if r.id == rid)
            prev_action_score += rv / cov

    return new_action_score - prev_action_score


def optimistic_utility(agent, new_action, coverage_map, M, alpha=DEFAULT_OPTIMISTIC_ALPHA):
    """
    Agents receive value for a resource if the exact max_cover is achieved or exceeded, however, they are minimally
    incentivized to move to high value uncovered resources at a constant alpha value.
        If coverage >= M, 1/num_agents reward.
        If coverage < M, (resource.value*alpha) reward.

    :param agent: current 'awake' agent
    :param new_action: candidate action to be scored
    :param coverage_map: current state of the system
    :param M: max_cover
    :param alpha: incentive to move to an uncovered resource
    :return: difference between the agents current action, and the candidate action
    """
    current_action = agent.current_action
    previous_coverage = filter_coverage_map(coverage_map, current_action)

    old_ids = {r.id for r in current_action}
    new_ids = {r.id for r in new_action}

    new_action_score = 0
    prev_action_score = 0

    # score resources newly added
    for rid in new_ids - old_ids:
        cov = previous_coverage.get(rid, 0) + 1
        rv = next(r.value for r in new_action if r.id == rid)
        if cov == 1:
            new_action_score += rv * alpha
        elif cov >= M:
            new_action_score += rv / cov

    # subtract score for resources removed
    for rid in old_ids - new_ids:
        cov = coverage_map.get(rid, 0)
        rv = next(r.value for r in current_action if r.id == rid)
        if cov == 1:
            prev_action_score += rv * alpha
        elif cov >= M:
            prev_action_score += rv / cov

    return new_action_score - prev_action_score


def filter_coverage_map(coverage_map, action):
    """
    Get copy of coverage_map without agent current action coverage.

    :param coverage_map: current state of the system
    :param action: agents currently selected action
    :return: coverage map if the agent is performing no actions
    """
    map_copy = deepcopy(coverage_map)
    action_ids = {r.id for r in action}
    for rid in action_ids:
        map_copy[rid] = max(map_copy.get(rid, 0) - 1, 0)
    return map_copy
