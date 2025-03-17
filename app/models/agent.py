class Agent:
    """
    Simple class representing an agent.

    Attributes:
        id: integer value representing the agent id.
        action_set: list of actions representing resources this agent can cover.
        current_action: current coverage of the agent
    """

    def __init__(self, agent_id, action_set, utility):
        self.id = agent_id
        self.action_set = action_set
        self.current_action = set()
        self.utility_function = utility

    def evaluate_action(self, action, system, utility_function):
        return utility_function(self, action, system.resource_coverage, system.M)
