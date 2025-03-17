import inspect
import random
import time
import uuid

import numpy as np

from app.core.algorithms import function_map
from app.core.data_collector import DataCollector
from app.core.system import System
from app.core.utils import marginal_contribution_utility, equal_share_utility, optimistic_utility
from app.models.agent import Agent
from app.models.resource import Resource
from app.utils.common_utils import export_scenario_to_json
from app.utils.common_utils import generate_param_analysis_plot, generate_histogram_analysis_plot
from app.utils.constants import *


def generate_problem_instance(num_resources, num_agents, action_size_range,
                              action_subset_size_range, m, resource_val_range, num_trials, util_func):
    """
    Create the System.

    :param num_resources: number of resources to add to system
    :param num_agents: number of agents to add to system
    :param action_size_range: range for size of each action set
    :param action_subset_size_range: range for size of each subset of resources
    :param m: maximum cover
    :param resource_val_range: range for resource value
    :param num_trials: number of trials, need to create a system object per trial
    :param util_func: utility of the agents
    :return: system_dict: dictionary of trial_num -> system object
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

            utility = ""
            if util_func == MC_UTILITY:
                utility = marginal_contribution_utility
            if util_func == ES_UTILITY:
                utility = equal_share_utility
            if util_func == OPTIMISTIC_UTILITY:
                utility = optimistic_utility

            agents.append(Agent(i, action_set, utility))

        system_dict[trial] = System(resources, agents, m)
    return system_dict


# TODO:: implement
def run_from_json(args):
    return 0


def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """

    analyze_beta = args.analyze_beta
    analyze_temp = args.analyze_temperature

    if analyze_beta and analyze_temp:
        conduct_parameter_analysis(args, b=True, temp=True)
        return

    if analyze_beta:
        conduct_parameter_analysis(args, b=True)
        return

    if analyze_temp:
        conduct_parameter_analysis(args, temp=True)
        return

    algorithm = args.algorithm
    distributions = args.distribution.split(',')
    if algorithm != PROB_RESPONSE:  # distributions only apply to prob response algo
        distributions = []
    iter_per_trial = args.iterations_per_trial
    beta = args.beta
    temperature = args.temperature
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
    util_func = args.utility_function

    system_dict = generate_problem_instance(num_resources, num_agents,
                                            (agent_action_len_lb, agent_action_len_ub),
                                            (agent_subset_len_lb, agent_subset_len_ub),
                                            m, (resource_val_lb, resource_val_ub), num_trials, util_func)

    # set up data collector
    algo_dist_names = [f"{algorithm}:{dist}" for dist in distributions]
    data_collector = DataCollector(algorithms=algo_dist_names,
                                   uuid_file_map={algo_dist: str(uuid.uuid4()) for algo_dist in algo_dist_names})
    saved_system_file = {}
    algo_func_args = {algo: {} for algo in algo_dist_names}
    keys_to_exclude = ["system", "data_collector"]  # dont include non-serializable data

    for trial, system in system_dict.items():
        print("Calculating system optimal score")
        start_t = time.time()
        optimal_score = function_map.get(BRUTE_FORCE)(system)
        end_t = time.time()
        print(
            f"Optimal System score for this configuration {optimal_score:.3f}, calculated in {end_t - start_t:.3f} seconds")

        # save an instance of the system object to json, file uuid stored in run summary
        saved_system_file[trial] = export_scenario_to_json(system, JSON_SAVE_PATH)
        for dist in distributions:
            _, func_args = call_target_algorithm(algorithm=algorithm, distribution=dist, system=system,
                                                 max_iterations=iter_per_trial,
                                                 beta=beta, data_collector=data_collector,
                                                 trial_num=trial, conv_iter=sys_convergence, temperature=temperature,
                                                 population_size=population_size,
                                                 mutation_rate=mutation_rate, tournament_k=tournament_k,
                                                 num_parents=num_parents,
                                                 generational_size=generational_size, k_crossover=k_crossover)

            algo_func_args[f"{algorithm}:{dist}"] = {k: v for k, v in func_args.items() if k not in keys_to_exclude}

        data_collector.summarize_results(saved_system_file, algo_func_args, optimal_score)


def conduct_parameter_analysis(args, b=None, temp=None):
    """
    Parse arguments from CMD or app.props, create a system, run targeted parameter analysis.

    :param args: command line arguments, or default values from application.properties
    :param b: is this a beta experiment
    :param temp: is this a temperature experiment
    :return: none
    """
    beta = args.beta
    temperature = args.temperature

    algorithm = PROB_RESPONSE
    distribution = []
    beta_vals = []
    temperature_vals = []
    param_score_history = {}

    if b is not None:
        distribution.append(APPROX_BEST_RESPONSE)
        beta_vals = np.arange(beta, MAX_BETA + BETA_STEP_SIZE, BETA_STEP_SIZE)
        for beta in beta_vals:
            param_score_history.setdefault(f"b{beta:.3f}", [])
    if temp is not None:
        distribution.append(LOGIT_RESPONSE)
        temperature_vals = np.logspace(
            np.log10(temperature), np.log10(MAX_TEMP),
            num=int(np.log2(MAX_TEMP / temperature)) + 1,
            base=10
        )
        for t in temperature_vals:
            param_score_history.setdefault(f"t{t:.3f}", [])

    num_trials = max(len(temperature_vals), len(beta_vals))

    if b is None:
        beta_vals = np.zeros(num_trials)

    if temp is None:
        temperature_vals = np.zeros(num_trials)

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
        sys_convergence = args.system_convergence_iter
        trial_repetitions = args.trial_repetitions
        util_func = args.utility_function

        # only one system configuration for this analysis
        system = generate_problem_instance(num_resources, num_agents,
                                           (agent_action_len_lb, agent_action_len_ub),
                                           (agent_subset_len_lb, agent_subset_len_ub),
                                           m, (resource_val_lb, resource_val_ub), 1, util_func)[0]  # pull dict val

        save_file = export_scenario_to_json(system, JSON_SAVE_PATH)
        print("Calculating system optimal score")
        start_t = time.time()
        optimal_score = function_map.get(BRUTE_FORCE)(system)
        end_t = time.time()
        print(
            f"Optimal System score for this configuration {optimal_score:.3f}, calculated in {end_t - start_t:.3f} seconds")
        saved_system_file = {}

        keys_to_exclude = ["system", "data_collector"]  # dont include non-serializable data
        for dist in distribution:
            score_history = []
            data_key = f"{algorithm}:{dist}"
            data_collector = DataCollector(algorithms=[data_key], uuid_file_map={data_key: str(uuid.uuid4())})
            for trial in range(num_trials):
                repetition_scores = []
                if trial >= len(beta_vals) and dist == APPROX_BEST_RESPONSE:
                    continue
                if trial >= len(temperature_vals) and dist == LOGIT_RESPONSE:
                    continue
                # save an instance of the system object to json, file uuid stored in run summary
                saved_system_file[trial] = save_file
                for repetition in range(trial_repetitions):
                    score, func_args = call_target_algorithm(algorithm=algorithm, distribution=dist, system=system,
                                                             max_iterations=iter_per_trial,
                                                             beta=beta_vals[min(trial, len(beta_vals) - 1)],
                                                             data_collector=data_collector,
                                                             trial_num=trial, conv_iter=sys_convergence,
                                                             temperature=temperature_vals[
                                                                 min(trial, len(temperature_vals) - 1)])
                    repetition_scores.append(score)

                    beta_key = f"b{beta_vals[min(trial, len(beta_vals) - 1)]:.3f}"
                    temp_key = f"t{temperature_vals[min(trial, len(temperature_vals) - 1)]:.3f}"

                    if beta_key in param_score_history and dist == APPROX_BEST_RESPONSE:
                        param_score_history[beta_key].append(score)
                    if temp_key in param_score_history and dist == LOGIT_RESPONSE:
                        param_score_history[temp_key].append(score)

                    func_args = {k: v for k, v in func_args.items() if k not in keys_to_exclude}
                    data_collector.summarize_results(saved_system_file, {data_key: func_args}, optimal_score)

                score_history.append(np.mean(repetition_scores))

            # Averaged over all repetitions
            if dist == APPROX_BEST_RESPONSE:
                generate_param_analysis_plot(beta_vals, score_history, BETA, optimal_score)
                generate_histogram_analysis_plot(param_score_history, BETA, HIST_BINS, optimal_score)
                score_history = []
            if dist == LOGIT_RESPONSE:
                generate_param_analysis_plot(temperature_vals, score_history, TEMP, optimal_score)
                generate_histogram_analysis_plot(param_score_history, TEMP, HIST_BINS, optimal_score)


def call_target_algorithm(algorithm, **kwargs):
    """
    Parse any needed parameters, call the target algorithm passing needed parameters.

    :param algorithm: target algorithm to invoke, holds the name of the algorithm
    :param kwargs: additional arguments to pass to the algorithm
    :return: score of system after running target algorithm
             filtered_args: arguments to pass to the target algorithm for summary
    """
    algo = function_map.get(algorithm)
    if not algo:
        raise ValueError(f"Algorithm '{algorithm}' not found in function_map")

    sig = inspect.signature(algo)
    filtered_args = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return algo(**filtered_args), filtered_args
