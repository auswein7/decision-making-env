def system_utility(agent, action, system):
    agent.current_action = action
    return system.system_score()
