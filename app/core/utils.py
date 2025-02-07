def system_utility(agent, action, system):
    """
    Test the system score with a potential agent action

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
