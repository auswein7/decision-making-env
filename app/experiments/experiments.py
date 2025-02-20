import random
import inspect

import numpy as np
import uuid

from app.models.resource import Resource
from app.models.agent import Agent

from app.core.system import System
from app.core.algorithms import function_map
from app.core.utils import system_utility
from app.core.data_collector import DataCollector

from app.utils.common_utils import load_scenario_from_json
from app.utils.common_utils import export_scenario_to_json
from app.utils.common_utils import generate_param_analysis_plot
from app.utils.constants import JSON_SAVE_PATH, JSON_LOAD_PATH, APPROX_BEST_RESPONSE, MAX_BETA, BETA_STEP_SIZE, \
    LOGIT_RESPONSE, MAX_TEMP, TEMP_STEP_SIZE, BRUTE_FORCE


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


# TODO:: run from json when conducting beta trial
# TODO:: run from json with newly added params, make sure it works for all algorithms
def run_from_json(args):
    algorithms = args.algorithm.split(',')
    iter_per_trial = args.iterations_per_trial
    beta = args.beta
    temperature = args.temperature
    sys_convergence = args.system_convergence_iter
    generate_graphics = args.generate_graphics

    # set up data collector
    data_collector = DataCollector(algorithms=algorithms, uuid_file_map={algo: str(uuid.uuid4()) for algo in algorithms})

    system = load_scenario_from_json(JSON_LOAD_PATH)
    optimal_score, _ = function_map.get(BRUTE_FORCE)(system)

    algo_func_args = {algo: {} for algo in algorithms}
    keys_to_exclude = ["system", "data_collector"]
    sim_json = export_scenario_to_json(system, JSON_SAVE_PATH)
    for algorithm in algorithms:
        _, func_args = call_target_algorithm(algorithm=algorithm, system=system, max_iterations=iter_per_trial, beta=beta,
                              generate_graphics=generate_graphics, data_collector=data_collector, trial_num=1,
                              cov_iter=sys_convergence,
                              temperature=temperature)



        algo_func_args[algorithm] = {k: v for k, v in func_args.items() if k not in keys_to_exclude}

    data_collector.summarize_results([sim_json], algo_func_args, optimal_score)


def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """

    analyze_beta = args.analyze_beta
    analyze_temp = args.analyze_temperature

    if analyze_beta:
        conduct_parameter_analysis(args, "beta")
        return

    if analyze_temp:
        conduct_parameter_analysis(args, "temp")
        return

    algorithms = args.algorithm.split(',')
    iter_per_trial = args.iterations_per_trial
    beta = args.beta
    temperature = args.temperature
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
    population_size = args.population_size
    mutation_rate = args.mutation_rate
    tournament_k = args.tournament_k
    num_parents = args.num_parents
    generational_size = args.generational_size
    k_crossover = args.k_crossover

    system_dict = generate_problem_instance(num_resources, num_agents,
                                            (agent_action_len_lb, agent_action_len_ub),
                                            (agent_subset_len_lb, agent_subset_len_ub),
                                            m, (resource_val_lb, resource_val_ub), num_trials)

    # set up data collector
    data_collector = DataCollector(algorithms=algorithms, uuid_file_map={algo: str(uuid.uuid4()) for algo in algorithms})
    save_file_per_trial = {}
    algo_func_args = {algo:{} for algo in algorithms}
    keys_to_exclude = ["system", "data_collector"]

    for trial, system in system_dict.items():
        optimal_score, _ = function_map.get(BRUTE_FORCE)(system)
        save_file_per_trial[trial] = export_scenario_to_json(system, JSON_SAVE_PATH)
        for algorithm in algorithms:
            _, func_args = call_target_algorithm(algorithm=algorithm, system=system, max_iterations=iter_per_trial, beta=beta,
                                  generate_graphics=generate_graphics, data_collector=data_collector,
                                  trial_num=trial, conv_iter=sys_convergence, temperature=temperature,
                                  population_size=population_size,
                                  mutation_rate=mutation_rate, tournament_k=tournament_k, num_parents=num_parents,
                                  generational_size=generational_size, k_crossover=k_crossover)

            algo_func_args[algorithm] = {k: v for k, v in func_args.items() if k not in keys_to_exclude}

        data_collector.summarize_results(save_file_per_trial, algo_func_args, optimal_score)


def conduct_parameter_analysis(args, param_name):
    algorithm = ""
    beta_vals = []
    temperature_vals = []

    num_trials = args.num_trials
    beta = args.beta
    temperature = args.temperature

    if param_name == "beta":
        algorithm = APPROX_BEST_RESPONSE
        beta_vals = np.arange(beta, MAX_BETA + BETA_STEP_SIZE, BETA_STEP_SIZE)
        num_trials = len(beta_vals)
        temperature_vals = np.zeros(num_trials)
    if param_name == "temp":
        algorithm = LOGIT_RESPONSE
        temperature_vals = np.arange(temperature, MAX_TEMP + TEMP_STEP_SIZE, TEMP_STEP_SIZE)
        num_trials = len(temperature_vals)
        beta_vals = np.zeros(num_trials)
    if algorithm != "":
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
        generate_graphics = args.generate_graphics
        sys_convergence = args.system_convergence_iter

        # only one system configuration for this analysis
        system = generate_problem_instance(num_resources, num_agents,
                                           (agent_action_len_lb, agent_action_len_ub),
                                           (agent_subset_len_lb, agent_subset_len_ub),
                                           m, (resource_val_lb, resource_val_ub), 1)[0]

        data_collector = DataCollector(algorithms=[algorithm], uuid_file_map={algorithm: str(uuid.uuid4())})
        save_file = export_scenario_to_json(system, JSON_SAVE_PATH)

        score_history = []
        optimal_score, _ = function_map.get(BRUTE_FORCE)(system)
        save_file_per_trial = {}
        keys_to_exclude = ["system", "data_collector"]

        for trial in range(num_trials):
            save_file_per_trial[trial] = save_file
            score, func_args = call_target_algorithm(algorithm=algorithm, system=system, max_iterations=iter_per_trial, beta=beta_vals[trial],
                                          generate_graphics=generate_graphics, data_collector=data_collector,
                                          trial_num=trial, conv_iter=sys_convergence, temperature=temperature_vals[trial])
            score_history.append(score)

            func_args = {k: v for k, v in func_args.items() if k not in keys_to_exclude}
            data_collector.summarize_results(save_file_per_trial, {algorithm:func_args}, optimal_score)

        if algorithm == APPROX_BEST_RESPONSE:
            generate_param_analysis_plot(beta_vals, score_history, "Beta")
        if algorithm == LOGIT_RESPONSE:
            generate_param_analysis_plot(temperature_vals, score_history, "Temperature")




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

    return algo(**filtered_args), filtered_args
