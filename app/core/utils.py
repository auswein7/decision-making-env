def global_visibility_utility(agent, action, system):
    """
    Test the system score with a potential agent action, agents can see the score of the overall system.

    :param agent: current 'awake' agent
    :param action: the current candidate action for the agent
    :param system: current problem layout
    :return: the score of the system if this action is taken
    """
    previous_action = agent.current_action
    agent.current_action = action

    sys_score = system.system_score()

    agent.current_action = previous_action
    return sys_score


def local_visibility_utility(agent, action):
    """
    Test the system score with a potential agent action, agents can not see the score of
    the overall system.

    :param agent: current 'awake' agent
    :param action: the current candidate action for the agent
    :return: the score of the system if this action is taken
    """
    #TODO:: IMPLEMENT
    return 0