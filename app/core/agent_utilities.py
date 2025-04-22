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
    new_action_score = 0
    prev_action_score = 0

    previous_coverage = filter_coverage_map(coverage_map, current_action)

    for curr_resource, past_resource in zip(new_action, current_action):
        curr_resource_id = curr_resource.id
        curr_resource_value = curr_resource.value

        num_agents = previous_coverage.get(curr_resource_id, 0) + 1
        if num_agents > M:
            new_action_score = 0
        if num_agents == M:
            new_action_score += curr_resource_value

        prev_resource_id = past_resource.id
        prev_resource_value = past_resource.value

        num_agents = coverage_map.get(prev_resource_id, 0)
        if num_agents > M:
            prev_action_score = 0
        if num_agents == M:
            prev_action_score += prev_resource_value

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
    new_action_score = 0
    prev_action_score = 0

    previous_coverage = filter_coverage_map(coverage_map, current_action)

    for curr_resource, past_resource in zip(new_action, current_action):
        curr_resource_id = curr_resource.id
        curr_resource_value = curr_resource.value

        num_agents = previous_coverage.get(curr_resource_id, 0) + 1
        if num_agents >= M:
            new_action_score += curr_resource_value / num_agents

        prev_resource_id = past_resource.id
        prev_resource_value = past_resource.value

        num_agents = coverage_map.get(prev_resource_id, 0)
        if num_agents >= M:
            prev_action_score += prev_resource_value / num_agents

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
    new_action_score = 0
    prev_action_score = 0

    previous_coverage = filter_coverage_map(coverage_map, current_action)

    for curr_resource, past_resource in zip(new_action, current_action):
        curr_resource_id = curr_resource.id
        curr_resource_value = curr_resource.value

        num_agents = previous_coverage.get(curr_resource_id, 0) + 1
        if num_agents == 1:
            new_action_score = curr_resource_value * alpha
        if num_agents >= M:
            new_action_score += curr_resource_value / num_agents

        prev_resource_id = past_resource.id
        prev_resource_value = past_resource.value

        num_agents = coverage_map.get(prev_resource_id, 0)
        if num_agents == 1:
            prev_action_score = prev_resource_value * alpha
        if num_agents >= M:
            prev_action_score += prev_resource_value / num_agents

    return new_action_score - prev_action_score


def filter_coverage_map(coverage_map, action):
    """
    Get copy of coverage_map without agent current action coverage.

    :param coverage_map: current state of the system
    :param action: agents currently selected action
    :return: coverage map if the agent is performing no actions
    """
    map_copy = deepcopy(coverage_map)
    action_ids = [resource.id for resource in action]
    for resource_id, _ in map_copy.items():
        if resource_id in action_ids:
            map_copy[resource_id] -= 1
    return map_copy
