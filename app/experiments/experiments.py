import random
from app.models.resource import Resource
from app.models.agent import Agent
from app.core.system import System
from app.core.algorithms import best_response
from app.core.utils import system_utility
from app.core.utils import log_system_properties
from app.core.utils import log_agent_allocation
from app.utils.logger import Logger

logger = Logger.get_logger()

def generate_problem_instance(num_resources, num_agents, action_size_range, m):
    resources = [Resource(i, random.randint(1, 10)) for i in range(num_resources)]
    agents = []

    for i in range(num_agents):
        action_set = set()
        ##TODO::Make agent action count passed by properties file
        for _ in range(random.randint(2, 3)):
            action = set(random.sample(resources, random.randint(*action_size_range)))
            action_set.add(frozenset(action))
        agents.append(Agent(i, action_set))

    return System(resources, agents, m, system_utility)


def run_experiments(num_trials, num_resources, num_agents, m):
    results = []
    for trial_num in range(num_trials):
        system = generate_problem_instance(num_resources, num_agents, (1, 4), m)
        log_system_properties(system, trial_num)
        best_response(system)
        results.append(system.system_score())
        log_agent_allocation(system)

    logger.info(f"Average System Score: {sum(results) / num_trials}")




