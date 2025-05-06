class System:
    """
    A class representing a System.

    Attributes:
        M: max_cover
        resource_coverage: current coverage for each resource
        score: current score
        id: uuid identifier
    """

    def __init__(self, resources, agents, m, id):
        self.resources = resources
        self.agents = agents
        self.M = m
        self.resource_coverage = {r.id: 0 for r in resources}
        self.score = 0
        self.id = id
        self.optimal_score = None
        self.optimal_coverage = None
        self.feasibility_margin = None
        self.resource_entropy = None
        self.overlap_density = None

    def system_score(self):
        """
        Evaluate the current system score dependent upon the current coverage map.

        :return: score: the score of the system
        """
        resource_coverage = {resource.id: 0 for resource in self.resources}

        for agent in self.agents:
            for resource in agent.current_action:
                resource_coverage[resource.id] += 1

        score = sum(resource.value for resource in self.resources if resource_coverage[resource.id] >= self.M)

        self.resource_coverage = resource_coverage
        self.score = score

        return score
