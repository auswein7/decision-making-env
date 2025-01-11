class System:
    def __init__(self, resources, agents, m, utility_function):
        self.resources = resources
        self.agents = agents
        self.M = m                 # Minimum agents needed to cover a resource
        self.utility_function = utility_function
        self.resource_coverage = set()

    def system_score(self):
        """Compute the system-level score."""
        resource_coverage = {resource.id: 0 for resource in self.resources}

        # Count resource coverage
        for agent in self.agents:
            for resource in agent.current_action:
                resource_coverage[resource.id] += 1

        # Compute score based on M
        score = sum(resource.value for resource in self.resources if resource_coverage[resource.id] >= self.M)

        self.resource_coverage = resource_coverage

        return score
