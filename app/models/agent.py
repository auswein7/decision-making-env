class Agent:
    def __init__(self, agent_id, action_set):
        self.id = agent_id
        self.action_set = action_set
        self.current_action = set()

    ##TODO:: weight the individual action of the agent as well, currently only accounting for system level scoring
    def evaluate_action(self, action, system, utility_function):
        return utility_function(self, action, system)