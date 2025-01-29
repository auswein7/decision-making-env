import random

from app.models.resource import Resource
from app.models.agent import Agent

from app.core.system import System
from app.core.algorithms import function_map
from app.core.algorithms import brute_force
from app.core.utils import system_utility

from app.utils.common_utils import load_scenario_from_json
from app.utils.common_utils import log_system_properties
from app.utils.common_utils import log_agent_allocation
from app.utils.common_utils import export_scenario_to_json
from app.utils.logger import Logger

#TODO: Make this more permanent, maybe add a consts file
JSON_LOAD_PATH = "scenarios/test_scenario.json"
JSON_SAVE_PATH = "app/out"
logger = Logger.get_logger()

def generate_problem_instance(num_resources, num_agents, action_size_range,
                              action_subset_size_range, m, resource_val_range):
    """
    Create resources, agents, and System.


    :param num_resources: number of resources to add to system
    :param num_agents: number of agents to add to system
    :param action_size_range: range for size of each action set
    :param action_subset_size_range: range for size of each subset of resources
    :param m: maximum cover
    :param resource_val_range: range for resource value
    :return: newly created system given parameters
    """
    resources = [Resource(i, random.randint(*resource_val_range)) for i in range(num_resources)]
    agents = []

    for i in range(num_agents):
        action_set = set()
        # TODO:: Refactor, if an action subset is identical to one already in action_set, it will not be added
        while len(action_set) < random.randint(*action_size_range):
            action = set(random.sample(resources, random.randint(*action_subset_size_range)))
            action_set.add(frozenset(action))
        agents.append(Agent(i, action_set))

    return System(resources, agents, m, system_utility)

def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments, log info.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """
    load_from_config = bool(args.load_from_config)
    algorithm = args.algorithm
    save_threshold = args.save_threshold

    if load_from_config:
        system, algo_name = load_scenario_from_json(JSON_LOAD_PATH)
        brute_force_score = brute_force(system)[0]
        logger.info(f"Loaded configuration brute force score: {brute_force_score}")
        log_system_properties(system, 0)
        algo = function_map.get(algo_name)
        score = algo(system)
        if score > brute_force_score * save_threshold:
            export_scenario_to_json(system, algorithm, JSON_SAVE_PATH)
        log_agent_allocation(system)
        return

    # TODO:: clean this up, find a better way to pass in the params to this func
    num_trials = args.num_trials
    num_resources = args.num_resources
    num_agents = args.num_agents
    m = args.max_cover
    resource_val_lb = args.resource_val_lb
    resource_val_ub = args.resource_val_ub
    agent_action_len_lb = args.agent_action_len_lb
    agent_action_len_ub = args.agent_action_len_ub
    agent_subset_len_lb = args.agent_subset_len_lb
    agent_subset_len_ub = args.agent_subset_len_ub

    results = []
    for trial_num in range(num_trials):
        system = generate_problem_instance(num_resources, num_agents, (agent_action_len_lb, agent_action_len_ub),
                                           (agent_subset_len_lb, agent_subset_len_ub),
                                           m, (resource_val_lb, resource_val_ub))
        # TODO:: brute force takes far to long to run as agents and resource count increases
        # brute_force_score = brute_force(system)[0]
        brute_force_score = 0
        logger.info(f"Trial {trial_num} brute force score: {brute_force_score}")
        log_system_properties(system, trial_num)
        algo = function_map.get(algorithm)
        score = algo(system)
        if score > brute_force_score * save_threshold:
            export_scenario_to_json(system, algorithm, JSON_SAVE_PATH)
        results.append(system.system_score())
        log_agent_allocation(system)

    logger.info(f"Average System Score: {sum(results) / num_trials}")
