import random
import inspect

import numpy as np

from app.models.resource import Resource
from app.models.agent import Agent

from app.core.system import System
from app.core.algorithms import function_map
from app.core.utils import system_utility
from app.core.data_collector import DataCollector

from app.utils.common_utils import load_scenario_from_json
from app.utils.common_utils import export_scenario_to_json
from app.utils.common_utils import generate_beta_sys_score_plot
from app.utils.constants import JSON_SAVE_PATH, JSON_LOAD_PATH, APPROX_BEST_RESPONSE, MAX_BETA, BETA_STEP_SIZE

def generate_problem_instance(num_resources, num_agents, action_size_range,
                              action_subset_size_range, m, resource_val_range, num_trials):
    """
    Create resources, agents, and System.

    :param num_trials: number of trials, need to create a system object per trial
    :param num_resources: number of resources to add to system
    :param num_agents: number of agents to add to system
    :param action_size_range: range for size of each action set
    :param action_subset_size_range: range for size of each subset of resources
    :param m: maximum cover
    :param resource_val_range: range for resource value
    :return: newly created system given parameters
    """
    system_dict = {i: None for i in range(num_trials)}

    for trial in range(num_trials):
        resources = [Resource(i, random.randint(*resource_val_range)) for i in range(num_resources)]
        agents = []
        for i in range(num_agents):
            action_set = set()
            while len(action_set) < random.randint(*action_size_range):
                action = set(random.sample(resources, random.randint(*action_subset_size_range)))
                action_set.add(frozenset(action))
            agents.append(Agent(i, action_set))

        system_dict[trial] = System(resources, agents, m, system_utility)
    return system_dict


def run_from_json(args):
    algorithms = args.algorithm.split(',')
    iter_per_trial = args.iterations_per_trial
    beta = args.beta
    generate_graphics = args.generate_graphics

    # set up data collector
    data_collector = DataCollector(algorithms=algorithms)

    system, algo_name = load_scenario_from_json(JSON_LOAD_PATH)
    for algorithm in algorithms:
        call_target_algorithm(algorithm=algorithm, system=system, max_iterations=iter_per_trial, beta=beta,
                              generate_graphics=generate_graphics, data_collector=data_collector, trial_num=1)

        sim_json = export_scenario_to_json(system, JSON_SAVE_PATH)

        data_collector.summarize_results(saved_file_uuid=sim_json)
        data_collector.clear_data()


def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """

    analyze_beta = args.analyze_beta

    if analyze_beta:
        conduct_beta_analysis(args)

    algorithms = args.algorithm.split(',')
    iter_per_trial = args.iterations_per_trial
    beta = args.beta
    generate_graphics = args.generate_graphics
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
    sys_convergence = args.system_convergence_iter

    system_dict = generate_problem_instance(num_resources, num_agents,
                                            (agent_action_len_lb, agent_action_len_ub),
                                            (agent_subset_len_lb, agent_subset_len_ub),
                                            m, (resource_val_lb, resource_val_ub), num_trials)

    # set up data collector
    data_collector = DataCollector(algorithms=algorithms)
    save_file_per_trial = {}

    for trial, system in system_dict.items():
        save_file_per_trial[trial] = export_scenario_to_json(system, JSON_SAVE_PATH)
        for algorithm in algorithms:
            call_target_algorithm(algorithm=algorithm, system=system, max_iterations=iter_per_trial, beta=beta,
                                  generate_graphics=generate_graphics, data_collector=data_collector,
                                  trial_num=trial, conv_iter=sys_convergence)

    data_collector.summarize_results(save_file_per_trial)

def conduct_beta_analysis(args):
    algorithm = APPROX_BEST_RESPONSE
    starting_beta = args.beta
    beta_vals = np.arange(starting_beta, MAX_BETA + BETA_STEP_SIZE, BETA_STEP_SIZE)
    num_trials = len(beta_vals)

    iter_per_trial = args.iterations_per_trial
    num_resources = args.num_resources
    num_agents = args.num_agents
    m = args.max_cover
    resource_val_lb = args.resource_val_lb
    resource_val_ub = args.resource_val_ub
    agent_action_len_lb = args.agent_action_len_lb
    agent_action_len_ub = args.agent_action_len_ub
    agent_subset_len_lb = args.agent_subset_len_lb
    agent_subset_len_ub = args.agent_subset_len_ub

    # only one system configuration for this analysis
    system = generate_problem_instance(num_resources, num_agents,
                                            (agent_action_len_lb, agent_action_len_ub),
                                            (agent_subset_len_lb, agent_subset_len_ub),
                                            m, (resource_val_lb, resource_val_ub), 1)[0]

    data_collector = DataCollector(algorithms=[APPROX_BEST_RESPONSE])
    save_file = export_scenario_to_json(system, JSON_SAVE_PATH)

    score_history = []
    for trial in range(num_trials):
        score = function_map.get(algorithm)(system=system, max_iterations=iter_per_trial, beta=beta_vals[trial],
                                  generate_graphics=False, data_collector=data_collector,
                                  trial_num=trial)
        score_history.append(score)

    generate_beta_sys_score_plot(beta_vals, score_history)

    data_collector.summarize_results([save_file] * num_trials)



def call_target_algorithm(algorithm, **kwargs):
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
