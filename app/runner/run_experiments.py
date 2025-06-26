import random
import uuid

import numpy as np
import pandas as pd

from app.core.data_collector import DataCollector
from app.models.agent import Agent
from app.models.resource import Resource
from app.models.system import System
from app.runner.trial_runner import run_trial
from app.utils.common_utils import generate_param_analysis_plot, generate_zoomed_analysis_plot, \
    compute_system_difficulties, plot_scores_by_rank, get_iteration_range, plot_optimal_iterations, parse_score_history, \
    calc_and_time_optimal, load_scenario_from_json, init_random_actions, export_scenario_to_json, \
    plot_normalized_param_average, plot_difficulty_scatter, estimate_local_minimum
from app.utils.constants import *
from app.utils.graph_generator import generate_graphs


def generate_problem_instance(num_resources, num_agents, action_size_range,
                              action_subset_size_range, m, resource_val_range):
    """
    Create the System.

    :param num_resources: number of resources to add to system
    :param num_agents: number of agents to add to system
    :param action_size_range: range for size of each action set
    :param action_subset_size_range: range for size of each subset of resources
    :param m: maximum cover
    :param resource_val_range: range for resource values
    :return: system: single instance of a system
    """
    resources = [Resource(i, random.randint(*resource_val_range)) for i in range(num_resources)]
    agents = []
    for i in range(num_agents):
        action_set = []
        while len(action_set) < random.randint(*action_size_range):
            action = set(random.sample(resources, random.randint(*action_subset_size_range)))
            if action not in action_set:
                action_set.append(action)
        agents.append(Agent(i, action_set, None))
    sys = System(resources, agents, m, uuid.uuid4().hex)
    (sys.feasibility_margin, sys.resource_entropy, sys.overlap_density, sys.agent_heterogeneity, sys.agent_heterogeneity,
     sys.action_combinations) = compute_system_difficulties(systems=[sys])
    sys.optimal_score, sys.optimal_coverage = calc_and_time_optimal(system=sys, init_from_opt=False)
    sys.local_minima = estimate_local_minimum(system=sys)
    sys.generation_data = {"method": "random_gen"}
    return sys


def filter_run(args):
    """
    Parse arguments from CMD or app.props, create a system, run the runner.

    :param args: cmd args, or defaults from application.properties
    :return: none
    """
    if args.analyze_system:
        system_analysis_runner(args)
        return
    if args.find_optimal_iterations:
        optimal_iteration_runner(args)
        return
    if args.analyze_beta or args.analyze_temperature:
        parameter_analysis_runner(args, b=args.analyze_beta, temp=args.analyze_temperature)
        return
    if args.generate_graphs:
        generate_graphs(args)
        return


def optimal_iteration_runner(args):
    """
    Run trials to test different iteration counts for a set system, distribution, and utility function.

    Plots:
        X: the iteration count used for the trial
        Y: the box plot of trial_repetitions scores per trial
    """
    iteration_range = get_iteration_range(args.iterations_per_trial, OPT_ITER_STEP_SIZE)
    iter_data = {}
    if args.load_from_config:
        system = load_scenario_from_json(args.system_file_directory)
    else:
        system = [generate_problem_instance(
            args.num_resources, args.num_agents, (args.agent_action_len_lb, args.agent_action_len_ub),
            (args.agent_subset_len_lb, args.agent_subset_len_ub), args.max_cover,
            (args.resource_val_lb, args.resource_val_ub))]

    for sys in system:
        save_file = export_scenario_to_json(sys, JSON_SAVE_PATH) if not args.load_from_config else sys.id
        optimal_score = sys.optimal_score
        print(f"-------- System (ID={sys.id[:6]}) optimal score: {optimal_score} --------")
        if args.init_from_random:
            init_random_actions(system=sys)

        key = f"{sys.id}-{args.distribution}-{args.utility}"
        data_collector = DataCollector(data_key=[key], sim_sum_uuid={key: str(uuid.uuid4())})
        for trial, iter_range in enumerate(iteration_range):
            for repetition in range(args.trial_repetitions):
                score, func_args = run_trial(system=sys,
                                             max_iterations=iter_range,
                                             beta=args.beta,
                                             temperature=args.temperature,
                                             data_collector=data_collector, trial_num=iter_range,
                                             conv_iter=args.system_convergence_iter,
                                             agent_util=args.utility, data_key=key,
                                             distribution=args.distribution, output_dir=args.output_dir)

                data_collector.summarize_results(save_file, {key: func_args}, avg_score=score,
                                                 optimal_score=optimal_score)
                if iter_range in iter_data:
                    iter_data[iter_range].append(score)
                else:
                    iter_data[iter_range] = [score]

        plot_optimal_iterations(data=iter_data, sys_optimal=optimal_score, sys_id=sys.id, output_dir=args.output_dir)


def system_analysis_runner(args):
    """
    Run trials to test different system configurations for a set distribution and utility function.
    """
    if args.load_from_config:
        systems = load_scenario_from_json(args.system_file_directory)
        args.num_systems = len(systems)
    else:
        systems = [
            generate_problem_instance(
                num_resources=args.num_resources,
                num_agents=args.num_agents,
                action_size_range=(args.agent_action_len_lb, args.agent_action_len_ub),
                action_subset_size_range=(args.agent_subset_len_lb, args.agent_subset_len_ub),
                m=args.max_cover,
                resource_val_range=(args.resource_val_lb, args.resource_val_ub)
            )
            for _ in range(args.num_systems)
        ]

    fm_vals = {sys.id: sys.feasibility_margin for sys in systems}
    re_vals = {sys.id: sys.resource_entropy for sys in systems}
    od_vals = {sys.id: sys.overlap_density for sys in systems}
    ah_vals = {sys.id: sys.agent_heterogeneity for sys in systems}
    rh_vals = {sys.id: sys.resource_heterogeneity for sys in systems}
    ac_vals = {sys.id: sys.action_combinations for sys in systems}

    # collect trial scores per system
    scores_by_system = {sid: [] for sid in fm_vals}

    # Track optimal scores for box-plot
    sys_opts = {}

    for idx, system in enumerate(systems):
        key = f"{system.id}-{args.distribution}-{args.utility}"
        data_collector = DataCollector(data_key=[key], sim_sum_uuid={key: str(uuid.uuid4())})
        save_file = (
            export_scenario_to_json(system, JSON_SAVE_PATH)
            if not args.load_from_config else
            system.id
        )

        optimal_score = system.optimal_score
        sys_opts[system.id] = optimal_score
        print(f"-------- System {idx} (ID={system.id[:6]}) optimal score: {optimal_score} --------")

        if args.init_from_random:
            init_random_actions(system=system)

        for _ in range(args.trial_repetitions):
            score, func_args = run_trial(
                system=system,
                distribution=args.distribution,
                agent_util=args.utility,
                max_iterations=args.iterations_per_trial,
                beta=args.beta,
                temperature=args.temperature,
                data_collector=data_collector,
                trial_num=idx,
                conv_iter=args.system_convergence_iter,
                data_key=key,
                output_dir=args.output_dir
            )

            scores_by_system[system.id].append(score)

            data_collector.summarize_results(save_file, {key: func_args}, avg_score=score, optimal_score=optimal_score)

    avg_scores = {
        sid: (sum(lst) / len(lst)) if lst else 0.0
        for sid, lst in scores_by_system.items()
    }

    plot_difficulty_scatter(fm_vals, avg_scores, sys_opts, title="Feasibility Margin vs Avg Score",
                            xlabel="Feasibility Margin", out_dir=args.output_dir)
    plot_difficulty_scatter(re_vals, avg_scores, sys_opts, title="Resource Entropy vs Avg Score",
                            xlabel="Resource Entropy", out_dir=args.output_dir)
    plot_difficulty_scatter(od_vals, avg_scores, sys_opts, title="Overlap Density vs Avg Score",
                            xlabel="Overlap Density", out_dir=args.output_dir)
    plot_difficulty_scatter(ah_vals, avg_scores, sys_opts, title="Agent Heterogeneity vs Avg Score",
                            xlabel="Agent Heterogeneity", out_dir=args.output_dir)
    plot_difficulty_scatter(rh_vals, avg_scores, sys_opts, title="Resource Heterogeneity vs Avg Score",
                            xlabel="Resource Heterogeneity", out_dir=args.output_dir)
    plot_difficulty_scatter(ac_vals, avg_scores, sys_opts, title="Action Counts vs Avg Score",
                            xlabel="Action Counts", out_dir=args.output_dir)


def parameter_analysis_runner(args, b=None, temp=None):
    """
    Run trials to test different parameter values over n distributions and m utility functions with a set system.

    Plots:
        Zoomed Plots:
            X: target parameter values
            Y: the box plot of trial_repetitions scores per trial
        Param Analysis:
            X: target parameter values
            Y: the box plot of trial_repetitions averaged scores per trial
    """
    beta_vals = np.arange(args.beta, MAX_BETA + BETA_STEP_SIZE, BETA_STEP_SIZE) if b else np.array([])
    temperature_vals = np.logspace(
        np.log10(args.temperature), np.log10(MAX_TEMP),
        num=int(np.log2(MAX_TEMP / args.temperature)) + 1, base=10
    ) if temp else np.array([])

    if args.load_from_config:
        systems = load_scenario_from_json(args.system_file_directory)
    else:
        # generate exactly one system
        system = generate_problem_instance(
            args.num_resources,
            args.num_agents,
            (args.agent_action_len_lb, args.agent_action_len_ub),
            (args.agent_subset_len_lb, args.agent_subset_len_ub),
            args.max_cover,
            (args.resource_val_lb, args.resource_val_ub)
        )
        systems = [system]

    records = []
    opt_scores = {}
    for idx, system in enumerate(systems):
        optimal_score = system.optimal_score
        opt_scores[system.id] = optimal_score
        print(f"-------- System {idx} (ID={system.id[:6]}) optimal score: {optimal_score} --------")

        if args.init_from_random:
            init_random_actions(system=system)

        param_score_history = {
            (utility, BETA, float(beta)): []
            for utility in args.utility.split(",")
            for beta in beta_vals
        }
        param_score_history.update({
            (utility, TEMP, float(t)): []
            for utility in args.utility.split(",")
            for t in temperature_vals
        })

        num_trials = max(len(temperature_vals), len(beta_vals))
        beta_vals = beta_vals if beta_vals.size > 0 else np.zeros(num_trials)
        temperature_vals = temperature_vals if temperature_vals.size > 0 else np.zeros(num_trials)

        save_file = export_scenario_to_json(system, JSON_SAVE_PATH) if not args.load_from_config else system.id

        for utility in args.utility.split(","):
            for param_vals, param_label, distribution in [(beta_vals, BETA, APPROX_BEST_RESPONSE),
                                                          (temperature_vals, TEMP, LOGIT_RESPONSE)]:
                key = f"{system.id}-{distribution}-{utility}"
                data_collector = DataCollector(data_key=[key], sim_sum_uuid={key: str(uuid.uuid4())})
                if not param_vals.any():
                    continue

                for trial, param in enumerate(param_vals):
                    for repetition in range(args.trial_repetitions):
                        score, func_args = run_trial(system=system,
                                                     max_iterations=args.iterations_per_trial,
                                                     beta=param if param_label == BETA else None,
                                                     temperature=param if param_label == TEMP else None,
                                                     data_collector=data_collector, trial_num=trial,
                                                     conv_iter=args.system_convergence_iter,
                                                     agent_util=utility, data_key=key,
                                                     distribution=distribution, output_dir=args.output_dir)

                        param_score_history[(utility, param_label, float(param))].append(score)
                        data_collector.summarize_results(save_file, {key: func_args}, optimal_score, score)

                        # record for optimal param values plot
                        records.append({
                            'system_id': system.id,
                            'utility': utility,
                            'distribution': distribution,
                            'param_label': param_label,
                            'param_value': param,
                            'trial_idx': trial,
                            'repetition': repetition,
                            'score': score
                        })

        run_data = parse_score_history(param_score_history)
        generate_param_analysis_plot(data=run_data, sys_optimal=optimal_score, sys_id=system.id, output_dir=args.output_dir)
        generate_zoomed_analysis_plot(data=run_data, sys_optimal=optimal_score, sys_id=system.id, output_dir=args.output_dir)

    # generate plot of optimal temp and beta values
    plot_normalized_param_average(pd.DataFrame(records), optimal_scores=opt_scores, out_dir=args.output_dir)
