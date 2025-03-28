import random
import time
import uuid

import numpy as np

from app.core.algorithms import function_map
from app.core.data_collector import DataCollector
from app.core.system import System
from app.models.agent import Agent
from app.models.resource import Resource
from app.utils.common_utils import export_scenario_to_json
from app.utils.common_utils import generate_param_analysis_plot, generate_zoomed_analysis_plot
from app.utils.constants import *


def generate_problem_instance(num_resources, num_agents, action_size_range,
                              action_subset_size_range, m, resource_val_range):
    """
    Create the System.

    :param num_resources: number of resources to add to system
    :param num_agents: number of agents to add to system
    :param action_size_range: range for size of each action set
    :param action_subset_size_range: range for size of each subset of resources
    :param m: maximum cover
    :param resource_val_range: range for resource value
    :return: system: single instance of a system
    """
    resources = [Resource(i, random.randint(*resource_val_range)) for i in range(num_resources)]
    agents = []
    for i in range(num_agents):
        action_set = set()
        while len(action_set) < random.randint(*action_size_range):
            action = set(random.sample(resources, random.randint(*action_subset_size_range)))
            action_set.add(frozenset(action))

        # Utility function set by called algorithm
        agents.append(Agent(i, action_set, None))

    return System(resources, agents, m)


# TODO:: implement
def run_from_json(args):
    return 0


# TODO:: refactor to run over utility funcs
def run_experiments(args):
    """
    Parse arguments from CMD or app.props, create a system, run the experiments.

    :param args: command line arguments, or default values from application.properties
    :return: none
    """
    if args.analyze_beta or args.analyze_temperature:
        conduct_parameter_analysis(args, b=args.analyze_beta, temp=args.analyze_temperature)
        return


def conduct_parameter_analysis(args, b=None, temp=None):
    """
    Parse arguments from CMD or app.props, create a system, run targeted parameter analysis.

    :param args: command line arguments, or default values from application.properties
    :param b: is this a beta experiment
    :param temp: is this a temperature experiment
    :return: none
    """
    beta_vals = np.arange(args.beta, MAX_BETA + BETA_STEP_SIZE, BETA_STEP_SIZE) if b else np.array([])
    temperature_vals = np.logspace(
        np.log10(args.temperature), np.log10(MAX_TEMP),
        num=int(np.log2(MAX_TEMP / args.temperature)) + 1, base=10
    ) if temp else np.array([])

    param_score_history = {
        f"{utility},{BETA},{beta:.3f}": []
        for utility in args.utility_functions.split(",")
        for beta in beta_vals
    }
    param_score_history.update({
        f"{utility},{TEMP},{t:.3f}": []
        for utility in args.utility_functions.split(",")
        for t in temperature_vals
    })

    num_trials = max(len(temperature_vals), len(beta_vals))
    beta_vals = beta_vals if beta_vals.size > 0 else np.zeros(num_trials)
    temperature_vals = temperature_vals if temperature_vals.size > 0 else np.zeros(num_trials)

    # only one system configuration for this analysis
    system = generate_problem_instance(
        args.num_resources, args.num_agents, (args.agent_action_len_lb, args.agent_action_len_ub),
        (args.agent_subset_len_lb, args.agent_subset_len_ub), args.max_cover,
        (args.resource_val_lb, args.resource_val_ub))

    save_file = export_scenario_to_json(system, JSON_SAVE_PATH)
    print("Calculating system optimal score")
    start_t = time.time()
    optimal_score = function_map.get(BRUTE_FORCE)(system)
    end_t = time.time()
    print(
        f"Optimal System score for this configuration {optimal_score:.3f}, calculated in {end_t - start_t:.3f} seconds")
    for utility in args.utility_functions.split(","):
        print(f"STARTING RUNS FOR {utility}")
        for param_vals, param_label, distribution in [(beta_vals, BETA, APPROX_BEST_RESPONSE),
                                                      (temperature_vals, TEMP, LOGIT_RESPONSE)]:
            key = f"{distribution}-{utility}"
            data_collector = DataCollector(data_key=[key], uuid_file_map={key: str(uuid.uuid4())})
            if not param_vals.any():
                continue

            for trial, param in enumerate(param_vals):
                for repetition in range(args.trial_repetitions):
                    score, func_args = function_map.get(PROB_RESPONSE)(system=system,
                                                                       max_iterations=args.iterations_per_trial,
                                                                       beta=param if param_label == BETA else None,
                                                                       temperature=param if param_label == TEMP else None,
                                                                       data_collector=data_collector, trial_num=trial,
                                                                       conv_iter=args.system_convergence_iter,
                                                                       agent_util=utility, data_key=key,
                                                                       distribution=distribution)

                    param_score_history[f"{utility},{param_label},{param:.3f}"].append(score)
                    data_collector.summarize_results(save_file, {key: func_args}, optimal_score, score)

    run_data = parse_score_history(param_score_history)
    generate_param_analysis_plot(data=run_data, sys_optimal=optimal_score)
    generate_zoomed_analysis_plot(data=run_data, sys_optimal=optimal_score)


def parse_score_history(score_history):
    parsed_data = {}

    for data_key, scores in score_history.items():
        for param_type in [BETA, TEMP]:
            if param_type in data_key:
                utility = data_key.split(param_type)[0][:-1]
                param_value = float(data_key.replace(utility, '').replace(param_type, '').replace(',', ''))

                if (utility, param_type) not in parsed_data:
                    parsed_data[(utility, param_type)] = {'x': [], 'y': [], 'scores': []}

                parsed_data[(utility, param_type)]['x'].append(param_value)
                parsed_data[(utility, param_type)]['y'].append(np.mean(scores))
                parsed_data[(utility, param_type)]['scores'].append(scores)

    return parsed_data
