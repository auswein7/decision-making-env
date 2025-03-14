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
        #TODO:: for this score, score it for the improvement to the overall system score
        #TODO:: difference between past action, and the value that the new action will give to the system
        # TODO:: currently implemented is equal share, above is implementing marginal contribution
        num_agents = coverage_map.get(resource_id, 0) + 1

        if num_agents >= M:
            action_score += resource_value / num_agents

    agent.current_action = previous_action
    return action_score

# TODO:: add them valuing a resources being alone multiplied by a weighting factor to make it very small. So they
# TODO:: do not go away from already covering a resource with someone else. What if we set that param to not be close
# TODO:: to zero? Maybe we should entice the agents to leave the resources?