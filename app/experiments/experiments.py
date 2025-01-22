import random

from app.models.resource import Resource
from app.models.agent import Agent

from app.core.system import System
from app.core.algorithms import best_response
from app.core.utils import system_utility

from app.utils.common_utils import load_scenario_from_json
from app.utils.common_utils import log_system_properties
from app.utils.common_utils import log_agent_allocation
from app.utils.logger import Logger

#TODO: Make this more permanent, maybe add a consts file
JSON_PATH = "scenarios/test_scenario.json"
logger = Logger.get_logger()

def generate_problem_instance(num_resources, num_agents, action_size_range, m, resource_val_lb, resource_val_ub):
    """
    Create resources, agents, and System.

    :param num_resources: number of resources to add to system
    :param num_agents: number of agents to add to system
    :param action_size_range: range of values representing the size of each subset of resources
    :param m: maximum cover
    :param resource_val_lb: lower bound of the value of any resource
    :param resource_val_ub: upper bound of the value of any resource
    :return: newly created system given parameters
    """
    resources = [Resource(i, random.randint(resource_val_lb, resource_val_ub)) for i in range(num_resources)]
    agents = []

    for i in range(num_agents):
        action_set = set()
        for _ in range(random.randint(2, 3)):
            action = set(random.sample(resources, random.randint(*action_size_range)))
            action_set.add(frozenset(action))
        agents.append(Agent(i, action_set))

    return System(resources, agents, m, system_utility)

def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments, log info.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """
    num_trials = args.num_trials
    num_resources = args.num_resources
    num_agents = args.num_agents
    m = args.max_cover
    resource_val_lb = args.resource_val_lb
    resource_val_ub = args.resource_val_ub
    agent_subset_len_lb = args.agent_subset_len_lb
    agent_subset_len_ub = args.agent_subset_len_ub
    load_from_config = bool(args.load_from_config)

    if load_from_config:
        system = load_scenario_from_json(JSON_PATH)
        log_system_properties(system, 0)
        best_response(system)
        log_agent_allocation(system)
        return

    results = []
    for trial_num in range(num_trials):
        system = generate_problem_instance(num_resources, num_agents,
                                           (agent_subset_len_lb, agent_subset_len_ub),
                                           m, resource_val_lb, resource_val_ub)

        log_system_properties(system, trial_num)
        #TODO:: Will have to change once other algorithms are added
        best_response(system)
        results.append(system.system_score())
        log_agent_allocation(system)

    logger.info(f"Average System Score: {sum(results) / num_trials}")
