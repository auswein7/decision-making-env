def global_visibility_utility(agent, action, system):
    """
    Test the system score with a potential agent action, agents can see the score of the overall system.

    :param agent: current 'awake' agent
    :param action: the current candidate action for the agent
    :param system: current problem layout
    :return: the score of the system if this action is taken
    """
    previous_action = agent.current_action
    previous_coverage = system.resource_coverage

    agent.current_action = action
    sys_score = system.system_score()

    agent.current_action = previous_action
    system.resource_coverage = previous_coverage
    return sys_score


def local_visibility_utility(agent, action, coverage_map, M):
    """
    Test the system score with a potential agent action, agents can not see the score of
    the overall system. Agents receive proportional score if an agent is already covering.

    :param agent: current 'awake' agent
    :param action: the current candidate action for the agent
    :param coverage_map: current coverage in the system
    :param M: overall coverage
    :return: the score of the system if this action is taken
    """
    previous_action = agent.current_action
    agent.current_action = action

    # evaluate local visibility score, cannot invoke system.score()
    action_score = 0
    for resource in agent.current_action:
        resource_id = resource.id
        resource_value = resource.value
        num_agents = coverage_map.get(resource_id, 0) + 1

        if num_agents >= M:
            action_score += resource_value / num_agents

    agent.current_action = previous_action
    return action_score


def compute_action_value(action_set, coverage_map, M):
    total_value = 0

    for resource in action_set:
        resource_id = resource.id  # Assuming Resource object has an 'id' attribute
        resource_value = resource.value  # Assuming Resource object has a 'value' attribute

        num_agents = coverage_map.get(resource_id, 0) + 1  # Include the current agent

        if num_agents >= M:
            total_value += resource_value / num_agents

    return total_value