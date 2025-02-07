import random
import inspect

from app.models.resource import Resource
from app.models.agent import Agent

from app.core.system import System
from app.core.algorithms import function_map
from app.core.utils import system_utility
from app.core.data_collector import DataCollector

from app.utils.common_utils import load_scenario_from_json
from app.utils.common_utils import log_system_properties
from app.utils.common_utils import log_agent_allocation
from app.utils.common_utils import export_scenario_to_json
from app.utils.logger import Logger
from app.utils.constants import JSON_SAVE_PATH, JSON_LOAD_PATH

logger = Logger.get_logger()
data_collector = DataCollector()

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
        # TODO: if an action subset is identical to one already in action_set, it will not be added, refactor
        while len(action_set) < random.randint(*action_size_range):
            action = set(random.sample(resources, random.randint(*action_subset_size_range)))
            action_set.add(frozenset(action))
        agents.append(Agent(i, action_set))

    return System(resources, agents, m, system_utility)

# TODO:: pass in multiple algorithms through props, spin thread for each and run
def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments, log info.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """
    load_from_config = bool(args.load_from_config)
    algorithm = args.algorithm
    iter_per_trial = args.iterations_per_trial
    beta = args.beta
    generate_graphics = args.generate_graphics

    if load_from_config:
        system, algo_name = load_scenario_from_json(JSON_LOAD_PATH)
        log_system_properties(system, 0)

        call_algorithm(algorithm, system=system, max_iterations=iter_per_trial, beta=beta,
                       generate_graphics=generate_graphics, data_collector=data_collector, trial_num=1)

        sim_json = export_scenario_to_json(system, algorithm, JSON_SAVE_PATH)
        log_agent_allocation(system)

        data_collector.summarize_results(file_names=sim_json)
        return

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
    file_names = []
    for trial_num in range(num_trials):
        system = generate_problem_instance(num_resources, num_agents,
                                           (agent_action_len_lb, agent_action_len_ub),
                                           (agent_subset_len_lb, agent_subset_len_ub),
                                           m, (resource_val_lb, resource_val_ub))

        log_system_properties(system, trial_num)

        call_algorithm(algorithm, system=system, max_iterations=iter_per_trial, beta=beta,
                       generate_graphics=generate_graphics, data_collector=data_collector, trial_num=trial_num)
        
        file_names.append(export_scenario_to_json(system, algorithm, JSON_SAVE_PATH))
        results.append(system.system_score())
        log_agent_allocation(system)

    data_collector.summarize_results(file_names)
    logger.info(f"Average System Score: {sum(results) / num_trials}")

def call_algorithm(algorithm, **kwargs):
    """
    Parse any needed parameters, call the target algorithm passing needed parameters.

    :param algorithm: target algorithm to invoke, holds the name of the algorithm
    :param kwargs: additional arguments to pass to the algorithm
    :return: none
    """
    algo = function_map.get(algorithm)
    if not algo:
        raise ValueError(f"Algorithm '{algorithm}' not found in function_map")

    sig = inspect.signature(algo)
    filtered_args = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return algo(**filtered_args)