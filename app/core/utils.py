from copy import deepcopy

from app.utils.constants import DEFAULT_OPTIMISTIC_ALPHA


# TODO:: WHAT IF THE COVERAGE MAP IS THE SAME AGENT WE ARE EVALUATING


def marginal_contribution_utility(agent, new_action, coverage_map, M):
    previous_action = agent.current_action
    new_action_score = 0
    prev_action_score = 0

    previous_coverage = filter_coverage_map(coverage_map, previous_action)

    for curr_resource, past_resource in zip(new_action, previous_action):
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
    previous_action = agent.current_action
    new_action_score = 0
    prev_action_score = 0

    previous_coverage = filter_coverage_map(coverage_map, previous_action)

    for curr_resource, past_resource in zip(new_action, previous_action):
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
    previous_action = agent.current_action
    new_action_score = 0
    prev_action_score = 0

    previous_coverage = filter_coverage_map(coverage_map, previous_action)

    for curr_resource, past_resource in zip(new_action, previous_action):
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
    map_copy = deepcopy(coverage_map)
    action_ids = [resource.id for resource in action]
    for resource_id, _ in map_copy.items():
        if resource_id in action_ids:
            map_copy[resource_id] -= 1
    return map_copy
