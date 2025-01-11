class Agent:
    def __init__(self, agent_id, action_set):
        self.id = agent_id
        self.action_set = action_set  # A set of sets of resources
        self.current_action = set()  # Empty action to start

    def evaluate_action(self, action, system, utility_function):
        """Evaluate a given action using the utility function."""
        return utility_function(self, action, system)
