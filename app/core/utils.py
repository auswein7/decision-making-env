# add more utility functions to this file in the future to represent more advanced agent action allocation

def system_utility(agent, action, system):
    """
    Update the agent's current action with the current candidate action.

    :param agent: current 'awake' agent
    :param action: the current candidate action for the agent
    :param system: current problem layout
    :return: the score of the system after taking this action
    """
    agent.current_action = action
    return system.system_score()


