import random
from copy import deepcopy


class GeneticAlgorithm:

    def __init__(self, population_size, mutation_rate, k, num_parents, generational_size, k_cross):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.k = k
        self.num_parents = num_parents
        self.generational_size = generational_size
        self.k_cross = k_cross

        self.population = []

    def create_population(self, system):
        # set all agent actions to a random choice within their action set
        for _ in range(self.population_size):
            self.population.append(GeneticAlgorithm.select_random_agent_actions(deepcopy(system)))
        self.evaluate_fitness()

    @staticmethod
    def select_random_agent_actions(system):
        for agent in system.agents:
            agent.current_action = random.choice(list(agent.action_set))
        return system

    def evaluate_fitness(self):
        if not isinstance(self.population[0], tuple):
            for idx, system in enumerate(self.population):
                self.population[idx] = (system, system.system_score())

    def tournament_selection(self):
        # select the top ranked populations using tournament selection
        if self.k <= len(self.population):
            tournament_indices = random.sample(range(len(self.population)), self.k)
        else:
            tournament_indices = random.sample(range(len(self.population)), len(self.population))

        tournament = [(self.population[i][0], self.population[i][1]) for i in tournament_indices]
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def crossover(self, p1, p2):
        if self.k_cross < len(p1.agents):
            split_indices = random.sample(range(len(p1.agents)), self.k_cross)
            split_indices.sort()
            # create a child and cross actions from p1, p2
            child = deepcopy(p1)

            p1_agents = p1.agents
            p2_agents = p2.agents

            child_agents = []
            toggle = True  # select agents from p1 first
            prev_idx = 0

            for idx in split_indices:
                if toggle:
                    child_agents.extend(p1_agents[prev_idx:idx])
                else:
                    child_agents.extend(p2_agents[prev_idx:idx])
                toggle = not toggle
                prev_idx = idx

            if toggle:
                child_agents.extend(p1_agents[prev_idx:])
            else:
                child_agents.extend(p2_agents[prev_idx:])

            child.score = 0
            child.reset_coverage_map()
            return child
        else:
            print("k_cross can not be greater than agent count!")
            return -1

    def breed_population(self):
        if len(self.population) == 1:
            return False

        new_population = []
        new_pop_size = int(self.generational_size * self.population_size)

        while len(new_population) < new_pop_size:
            new_population.append(
                self.mutate_individual(self.crossover(self.tournament_selection(), self.tournament_selection())))

        self.population_size = new_pop_size
        self.population = new_population

    def mutate_individual(self, individual):
        # force an agent to choose a random action within their set based on mutation_rate
        for agent in individual.agents:
            if random.random() < self.mutation_rate:
                agent.current_action = random.choice(list(agent.action_set))

        return individual
