class System:
    """
    System objects are created per trial. Each system holds all needed data to represent a maximum n-times
    set cover problem.

    Attributes:
        resources: resources in the simulation
        agents: agents in the simulation
        M: maximum agent coverage needed to claim a resource
        resource_coverage: final coverage count of each resource
    """

    def __init__(self, resources, agents, m):
        self.resources = resources
        self.agents = agents
        self.M = m  # max-cover
        self.resource_coverage = set()
        self.score = 0

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
        self.score = score

        return score

    def reset_coverage_map(self):
        self.resource_coverage = set()
