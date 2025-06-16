import colorsys
import json
import math
import os
import random
import csv
import statistics
import time
from copy import deepcopy
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyvis.network import Network

from app.runner.trial_runner import run_trial
from app.core.algorithms import function_map
from app.models.agent import Agent
from app.models.resource import Resource
from app.models.system import System
from app.utils.constants import *

loaded_systems = []


def load_scenario_from_json(directory):
    """
    Load all system scenarios from JSON files in the given directory.

    :param directory: path to directory containing .json scenario files.
    :return: List[System] loaded systems.
    """
    systems = []
    seen_ids = set()

    # Iterate over every .json file in the directory
    for file_name in os.listdir(directory):
        if not file_name.lower().endswith('.json'):
            continue
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r') as f:
            data = json.load(f)

        sys_data = data.get("system", {})
        sys_id = sys_data.get("id")
        if not sys_id or sys_id in seen_ids:
            continue
        seen_ids.add(sys_id)

        resources = [Resource(r["id"], r["value"]) for r in data.get("resources", [])]
        id_to_res = {res.id: res for res in resources}

        agents = [Agent(a["id"], a.get("action_set", []), None) for a in data.get("agents", [])]

        for agent in agents:
            new_sets = []
            for subset in agent.action_set:
                # Keep only IDs present in resources
                valid = {id_to_res[rid] for rid in subset if rid in id_to_res}
                if valid:
                    new_sets.append(valid)
            agent.action_set = new_sets

        m = sys_data.get("m")
        sys = System(resources, agents, m, sys_id)

        sys.optimal_score = sys_data.get("optimal_score")
        sys.optimal_coverage = sys_data.get("optimal_coverage")
        sys.feasibility_margin = sys_data.get("feasibility_margin", {}).get(sys_id)
        sys.resource_entropy = sys_data.get("resource_entropy", {}).get(sys_id)
        sys.overlap_density = sys_data.get("overlap_density", {}).get(sys_id)
        sys.agent_heterogeneity = sys_data.get("agent_heterogeneity", {}).get(sys_id)
        sys.resource_heterogeneity = sys_data.get("resource_heterogeneity", {}).get(sys_id)
        sys.action_combinations = sys_data.get("action_combinations", {}).get(sys_id)
        sys.local_minima = sys_data.get("local_minima", {})
        sys.generation_data = sys_data.get("generation_data", {})

        systems.append(sys)

    return systems


def export_scenario_to_json(system=None, file_path="app/out"):
    """
    Export a given random system to json file to be reloaded in future runner.

    :param system: experiment system configuration
    :param file_path: path to scenario json file
    :return: filename: name of created scenario json file
    """
    generate_system_html(system, system.id, file_path)

    file_name = os.path.join(file_path, "saved_models", f"{system.id}.json")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    resources_data = [{"id": resource.id, "value": resource.value} for resource in system.resources]

    agents_data = [
        {
            "id": agent.id,
            "action_set": [
                [resource.id for resource in action]
                for action in agent.action_set
            ]
        }
        for agent in system.agents
    ]

    system_data = {
        "id": system.id,
        "m": system.M,
        "optimal_score": system.optimal_score,
        "optimal_coverage": system.optimal_coverage,
        "feasibility_margin": system.feasibility_margin,
        "overlap_density": system.overlap_density,
        "resource_entropy": system.resource_entropy,
        "agent_heterogeneity": system.agent_heterogeneity,
        "resource_heterogeneity": system.resource_heterogeneity,
        "action_combinations": system.action_combinations,
        "local_minima": system.local_minima,
        "generation_data": system.generation_data,
    }

    simulation_data = {
        "resources": resources_data,
        "agents": agents_data,
        "system": system_data,
    }

    with open(file_name, 'w') as json_file:
        json.dump(simulation_data, json_file, indent=4)

    return file_name


def generate_system_html(system, uuid_str, output_dir):
    file_name = os.path.join(output_dir, "saved_models", f"{uuid_str}.html")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    net = Network(
        height="5000px",
        width="5000px",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )

    net.set_options("""
    {
      "autoResize": false,
      "edges": {
        "color": { "highlight": "red" },
        "smooth": { "enabled": false }
      },
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 100
        },
        "barnesHut": {
          "gravitationalConstant": -200000,
          "centralGravity": 0.05,
          "springLength": 400,
          "springConstant": 0.02,
          "avoidOverlap": 1.5
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "highlightNearest": {
          "enabled": true,
          "degree": 1,
          "hover": false
        }
      }
    }
    """)

    # unique node color per agent
    n_agents = len(system.agents)
    agent_colors = {}
    for idx, agent in enumerate(system.agents):
        hue = idx / max(n_agents, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        agent_colors[agent.id] = hex_color

    # agent nodes
    for agent in system.agents:
        a_n = f"A{agent.id}"
        net.add_node(
            a_n,
            label=a_n,
            title=f"Agent {agent.id}",
            color="blue",
            shape="dot"
        )

    # add action nodes
    for agent in system.agents:
        base_n = f"A{agent.id}"
        for idx, subset in enumerate(agent.action_set):
            act_n = f"{base_n}_act{idx}"
            net.add_node(
                act_n,
                label=f"{agent.id}:{idx}",
                color=agent_colors[agent.id],
                shape="diamond"
            )
            net.add_edge(base_n, act_n)

    # resource nodes
    for resource in system.resources:
        r_n = f"R{resource.id}"
        net.add_node(
            r_n,
            label=r_n,
            title=f"Resource {resource.id} (Val: {resource.value})",
            color="green",
            shape="square"
        )

    # Action -> Resource edges
    for agent in system.agents:
        for idx, subset in enumerate(agent.action_set):
            act_n = f"A{agent.id}_act{idx}"
            for r in subset:
                net.add_edge(act_n, f"R{r.id}")

    net.write_html(file_name)
    with open(file_name, 'r') as f:
        html = f.read()

    # turn off the physics after it settles the nodes
    injection = """
    <script type="text/javascript">
      network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: { enabled: false } });
      });
    </script>
    </body>
    """
    html = html.replace("</body>", injection)
    with open(file_name, 'w') as f:
        f.write(html)


def generate_param_analysis_plot(data, sys_optimal, sys_id):
    """
    For all parameters, distributions, and utility functions used in a parameter analysis run. Render a box plot of the
    scores over all repetitions.

    :param data: formatted dictionary of run data
    :param sys_optimal: optimal score for the tested system
    :param sys_id: system.id uuid
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for (utility, var_name), values in data.items():
        folder_path = os.path.join(JSON_SAVE_PATH, f"{utility}-{var_name}-whisker")
        os.makedirs(folder_path, exist_ok=True)

        fallback_scores = load_best_observed_scores_from_log()
        if sys_optimal == -1 and fallback_scores is not None:
            sys_optimal = fallback_scores.get(sys_id, 1)

        normalized_scores = [np.array(scores) / sys_optimal for scores in values['scores']]

        plt.figure(figsize=(8, 5))
        plt.boxplot(normalized_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.xticks(ticks=range(1, len(values['x']) + 1), labels=[f"{x:.4f}" for x in values['x']])

        plt.ylim(0, 1)
        plt.xlabel(f"{var_name}", fontsize=8)
        plt.xticks(rotation=45)
        plt.ylabel("Normalized Score", fontsize=8)
        plt.title(f"{var_name} vs System Score ({utility})", fontsize=10)
        plt.grid(alpha=0.5)

        filename = os.path.join(folder_path, f"{sys_id}_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_scores_by_rank(data, title='', x_label='', y_label='Normalized System Scores', sys_opts=None):
    """
    For all systems create in a system analysis run. Generate a box plot of scores over all repetitions.

    :param data: formatted dictionary of run data
    :param title: plt title
    :param x_label: x_axis label
    :param sys_opts: optimal scores for all systems
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(JSON_SAVE_PATH, f"{title.replace(' ', '_').lower()}")
    os.makedirs(folder_path, exist_ok=True)

    sorted_items = sorted(data.items(), key=lambda x: x[1][0])

    x_labels = []
    box_data = []

    for system_id, values in sorted_items:
        rank = values[0]
        scores = values[1:]

        opt = sys_opts[system_id]

        # add due to change of not always calculating optimal sys scores
        fallback_scores = load_best_observed_scores_from_log()
        if opt == -1 and fallback_scores is not None:
            opt = fallback_scores.get(system_id, 1)

        if opt != 0:
            scores = [s / opt for s in scores]
        else:
            scores = [0 for _ in scores]

        short_id = system_id[:5]
        x_labels.append(f"{rank:.3f}\n{short_id}")
        box_data.append(scores)

    plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.xticks(ticks=range(1, len(x_labels) + 1), labels=x_labels, rotation=0)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    filename = os.path.join(folder_path, f"{x_label.replace(' ', '_').lower()}_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_normalized_param_average(df, optimal_scores, out_dir='app/out/opt_params'):
    """
    For each (distribution, utility) pair in `df`, compute the mean score at each parameter
    across all systems, normalize by each system’s optimal score, then average across systems
    and save a single plot of that curve.
    """
    os.makedirs(out_dir, exist_ok=True)

    fallback_scores = load_best_observed_scores_from_log()
    # Start with the original optimal scores
    patched_scores = {}
    for sys_id, opt in optimal_scores.items():
        if opt != -1 and opt > 0:
            patched_scores[sys_id] = opt
        elif fallback_scores and sys_id in fallback_scores and fallback_scores[sys_id] > 0:
            patched_scores[sys_id] = fallback_scores[sys_id]
        else:
            patched_scores[sys_id] = 1  # Avoid division by zero

    optimal_series = pd.Series(patched_scores)

    # one figure per (distribution, utility)
    for (dist, util), sub in df.groupby(['distribution', 'utility']):
        # average score per system & param
        avg = (
            sub
            .groupby(['system_id', 'param_label', 'param_value'])['score']
            .mean()
            .reset_index()
        )
        # pivot so rows = param_value, cols = system_id
        pivot = avg.pivot(index='param_value', columns='system_id', values='score')

        # normalize each column by that system’s optimal score
        pivot_norm = pivot.div(optimal_series, axis=1)

        # average across systems
        overall = pivot_norm.mean(axis=1)

        xlab = avg['param_label'].iat[0]
        file_name = f"{util}_{dist}_{xlab}.png"
        path = os.path.join(out_dir, file_name)

        plt.figure()
        plt.plot(overall.index.round(4), overall.values, marker='o')
        plt.xlabel(xlab)
        plt.ylabel('Normalized Average Score')
        plt.title(f"{util}–{dist}")
        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.savefig(path)
        plt.close()


def generate_zoomed_analysis_plot(data, sys_optimal, sys_id):
    """
    For all parameters, distributions, and utility functions used in a parameter analysis run. Render the scores
    attained during repetitions.

    :param data: formatted dictionary of run data
    :param sys_optimal: optimal score for the tested system
    :param sys_id: system.id uuid
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for (utility, var_name), values in data.items():
        folder_path = os.path.join(JSON_SAVE_PATH, f"{utility}-{var_name}-zoom")
        os.makedirs(folder_path, exist_ok=True)

        for param_value, scores in zip(values['x'], values['scores']):

            fallback_scores = load_best_observed_scores_from_log()
            if sys_optimal == -1 and fallback_scores is not None:
                sys_optimal = fallback_scores.get(sys_id, 1)

            normalized_scores = np.array(scores) / sys_optimal

            plt.figure(figsize=(8, 5))
            plt.ylim(0, 1)
            plt.plot(range(len(normalized_scores)), normalized_scores, marker='o', color="blue")
            plt.xlabel('Repetition', fontsize=8)
            plt.ylabel('Norm Score', fontsize=8)
            plt.title(f'{var_name}:{param_value:.4f} with {utility}', fontsize=10)
            plt.grid(alpha=0.5)

            filename = os.path.join(folder_path, f"{sys_id}_{param_value:.4f}_{timestamp}.png")
            plt.savefig(filename, dpi=300)
            plt.close()


def plot_difficulty_scatter(metric_map, avg_scores, optimal_scores, title, xlabel, out_dir='app/out/sys_scatters'):
    """
    Scatter plot of system difficulty vs. normalized average trial score
    """
    os.makedirs(out_dir, exist_ok=True)

    x = []
    y = []
    for sid, diff in metric_map.items():
        if sid in avg_scores and sid in optimal_scores and optimal_scores[sid] != 0:

            sys_optimal = optimal_scores[sid]
            fallback_scores = load_best_observed_scores_from_log()
            if sys_optimal == -1 and fallback_scores is not None:
                sys_optimal = fallback_scores.get(sid, 1)

            x.append(diff)
            y.append(avg_scores[sid] / sys_optimal)

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, s=60, edgecolors='black', linewidths=0.8, alpha=0.6, marker='o')

    plt.grid(alpha=0.5)
    plt.minorticks_on()

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Normalized System Score", fontsize=12)
    plt.title(title, fontsize=14, pad=10)

    plt.tight_layout()

    file_name = f"{xlabel.replace(' ', '_')}_vs_norm_score_scatter.png"
    plt.savefig(os.path.join(out_dir, file_name), dpi=150)
    plt.close()


def plot_optimal_iterations(data, sys_optimal, sys_id):
    """
    For all distribution iterations render the box plot score of the repetitions.

    :param data: formatted dictionary of run data
    :param sys_optimal: optimal score for the tested system
    :param sys_id: system.id uuid
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    title = "Normalized Scores vs Iterations"
    folder_path = os.path.join(JSON_SAVE_PATH, title.replace(" ", "_").lower())
    os.makedirs(folder_path, exist_ok=True)

    sorted_items = sorted(data.items())
    x_labels = [str(k) for k, _ in sorted_items]

    fallback_scores = load_best_observed_scores_from_log()
    if sys_optimal == -1 and fallback_scores is not None:
        sys_optimal = fallback_scores.get(sys_id, 1)

    box_data = []
    for _, scores in sorted_items:
        norm_scores = [s / sys_optimal for s in scores]
        box_data.append(norm_scores)

    plt.figure(figsize=(12, 6))
    plt.ylim(0, 1)
    plt.boxplot(box_data, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.xticks(ticks=range(1, len(x_labels) + 1), labels=x_labels)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Normalized Score")
    plt.grid(alpha=0.5)
    plt.tight_layout()

    filename = os.path.join(folder_path, f"{sys_id}_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()


def load_best_observed_scores_from_log(csv_path="app/out/system_trials.csv"):
    """
    Load the highest observed system score from trials log.
    Returns: dict mapping system_id -> highest observed score
    """
    df = pd.read_csv(csv_path)

    # Only keep meaningful scores
    df = df[df["observed_optimal"] > 0]

    # Group by system and find max
    best_scores = df.groupby("system_id")["observed_optimal"].max().to_dict()
    return best_scores


def format_agent_data(agents):
    """
    Convert agent action sets, and resource id within actions to parsable structure.

    :param agents: system agents
    :return: formatted agent data dictionary
    """
    out = {agent.id: [] for agent in agents}
    for agent in agents:
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                sub_list.append((resource.id, resource.value))
            out[agent.id].append(sub_list)
    return out


def compute_resource_value_heterogeneity(resources):
    """
    Compute the coefficient of variation (CV) in resource values.
    A value of 0 means all resources have the same value.

    :param resources: system resources
    :return: float indicating dispersion in resource values
    """
    values = [r.value for r in resources]

    if len(values) < 2:
        return 0.0

    mean = statistics.mean(values)
    std = statistics.pstdev(values)

    return std / mean if mean > 0 else 0.0


def compute_action_combinations(agents):
    """
    Compute how many unique agent action allocations the system has.

    :param agents: system agents
    :return: int number of unique agent action allocations
    """
    count = 1
    for agent in agents:
        count *= len(agent.action_set) if agent.action_set else 1
    return count


def compute_agent_action_heterogeneity(agents):
    """
    Compute how heterogeneous agents are in their # of actions and the avg size of those actions.
    Returns a value >=0 where 0 means perfectly uniform.
    """
    action_counts = [len(agent.action_set) for agent in agents]
    avg_action_sizes = []

    for agent in agents:
        sizes = [len(act) for act in agent.action_set]
        avg_action_sizes.append(statistics.mean(sizes) if sizes else 0.0)

    if len(action_counts) < 2:
        return 0.0

    def cv(vals):
        mew = statistics.mean(vals)
        theta = statistics.pstdev(vals)
        return (theta / mew) if mew > 0 else 0.0

    cv_counts = cv(action_counts)
    cv_sizes = cv(avg_action_sizes)

    # return average of two cv scores
    return (cv_counts + cv_sizes) / 2


def compute_overlap_density(agents, resources):
    """
    Compute the overlap density to quantify system difficulty.

    :param agents: system agents
    :param resources: system resources
    :return: overlap density of a system
    """
    reach_sets = []
    for agent in agents:
        covered = set()
        for action in agent.action_set:
            covered |= action
        reach_sets.append(covered)

    total_overlap = 0
    count = 0

    for i, j in combinations(range(len(agents)), 2):
        overlap = len(reach_sets[i] & reach_sets[j])
        total_overlap += overlap / len(resources)
        count += 1

    return total_overlap / count if count > 0 else 0


def compute_agent_resource_entropy(agents, resources):
    """
    Compute agent_resource_entropy to quantify system difficulty.

    :param agents: system agents
    :param resources: system resources
    :return: agent_resource_entropy of a system
    """
    resource_counts = {r: 0 for r in resources}
    total_hits = 0

    for agent in agents:
        for action in agent.action_set:
            for r in action:
                resource_counts[r] += 1
                total_hits += 1

    probs = [count / total_hits for count in resource_counts.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy


def compute_feasibility_margin(agents, resources, M):
    """
    Compute feasibility_margin to quantify system difficulty.

    :param agents: system agents
    :param resources: system resources
    :return: feasibility_margin of a system
    """
    resource_to_agents = {r: set() for r in resources}

    for agent in agents:
        for action in agent.action_set:
            for r in action:
                resource_to_agents[r].add(agent.id)

    total_margin = 0
    for r in resources:
        agent_count = len(resource_to_agents[r])
        margin = (agent_count - M) / M
        total_margin += margin

    return total_margin / len(resources)


def compute_system_difficulties(systems):
    """
    Compute difficulties for system across all metrics.

    :param systems: list of systems
    :return: dictionaries of system difficulties
    """
    system_difficulties_fm = {}
    system_difficulties_re = {}
    system_difficulties_od = {}
    system_difficulties_ah = {}
    system_difficulties_rh = {}
    system_difficulties_ac = {}

    for system in systems:
        system_difficulties_fm[system.id] = [compute_feasibility_margin(system.agents, system.resources, system.M)]
        system_difficulties_re[system.id] = [compute_agent_resource_entropy(system.agents, system.resources)]
        system_difficulties_od[system.id] = [compute_overlap_density(system.agents, system.resources)]
        system_difficulties_ah[system.id] = [compute_agent_action_heterogeneity(system.agents)]
        system_difficulties_rh[system.id] = [compute_resource_value_heterogeneity(system.resources)]
        system_difficulties_ac[system.id] = [compute_action_combinations(system.agents)]

    return (
        dict(sorted(system_difficulties_fm.items(), key=lambda x: x[1], reverse=True)),
        dict(sorted(system_difficulties_re.items(), key=lambda x: x[1], reverse=True)),
        dict(sorted(system_difficulties_od.items(), key=lambda x: x[1])),
        dict(sorted(system_difficulties_ah.items(), key=lambda x: x[1], reverse=True)),
        dict(sorted(system_difficulties_rh.items(), key=lambda x: x[1], reverse=True)),
        dict(sorted(system_difficulties_ac.items(), key=lambda x: x[1], reverse=True)),
    )


def get_iteration_range(initial_iters, relative_step):
    """
    Compute 4 values left of iterations, 5 values right of iterations, with a 10% step size

    :param initial_iters: iteration value passed in app.props
    :param relative_step: step size
    :return: list of iterations to test in convergence analysis
    """
    step = initial_iters * relative_step
    left = [int(initial_iters - step * i) for i in reversed(range(5))]  # center and 4 left
    right = [int(initial_iters + step * i) for i in range(1, 6)]  # 5 right
    return left + right


def parse_score_history(score_history):
    parsed_data = {}

    for (utility, param_label, param_value), scores in score_history.items():
        if not scores:
            print(f"[WARN] Skipping empty score list for {utility}, {param_label}, {param_value}")
            continue

        if (utility, param_label) not in parsed_data:
            parsed_data[(utility, param_label)] = {'x': [], 'y': [], 'scores': []}

        parsed_data[(utility, param_label)]['x'].append(param_value)
        parsed_data[(utility, param_label)]['y'].append(np.mean(scores))
        parsed_data[(utility, param_label)]['scores'].append(scores)

    return parsed_data


def calc_and_time_optimal(system, init_from_opt):
    """
    Calculate the system optimal score and output time taken to calculate.

    :param system: system to compute optimal score for.
    :param init_from_opt: clear the agent actions or not from brute force calc
    :return: optimal score of the system
    """
    if system.optimal_score is not None:
        print("System optimal score already calculated, returning...")
        return system.optimal_score, {}

    print("Calculating system optimal score")
    start_t = time.time()
    optimal_score, optimal_coverage = function_map.get(BRUTE_FORCE)(system, init_from_opt)
    end_t = time.time()
    print(
        f"Optimal System score for this configuration {optimal_score:.3f}, calculated in {end_t - start_t:.3f} seconds")
    system.optimal_score = optimal_score
    return optimal_score, optimal_coverage


def init_random_actions(system):
    for agent in system.agents:
        agent.current_action = random.choice(agent.action_set)


def calculate_system_convergence(score_history, curr_sys):
    """
    Compute system convergence based on stability of system score.

    :param score_history: Conv_iter previous system scores
    :param curr_sys: current system at this iteration
    :return: true if system has converged, false otherwise.
    """
    score_sim_count = 0
    score = curr_sys.score
    for idx, prev_score in enumerate(reversed(score_history)):
        if prev_score < score:
            return False
        else:
            score_sim_count += 1
        score = prev_score
    if score_sim_count >= len(score_history):
        return True
    return False


def estimate_local_minimum(system, num_iterations=6000):
    """
    Run one ABR and one Logit response trial to estimate a local minimum score.
    """
    local_min_scores = []

    for dist_name, param in [
        ("approximate_best_response", 1.0),
        ("logit_response", 0.000001)
    ]:
        score, _ = run_trial(
            system=deepcopy(system),
            distribution=dist_name,
            agent_util="marginal_contribution",
            max_iterations=num_iterations,
            beta=param,
            temperature=param,
            data_collector=None,
            trial_num=0,
            conv_iter=float("inf"),
            data_key="",
            computing_minima=True
        )
        local_min_scores.append(score)

    return min(local_min_scores)


def log_trial_to_dataset(system, score, distribution, beta, temperature, trial_num):
    output_path = JSON_SAVE_PATH + "/system_trials.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    local_min = system.local_minima
    param = beta if distribution == APPROX_BEST_RESPONSE else temperature

    graph_type = "N/A"
    graph_param = "N/A"
    generation_method = system.generation_data.get("method")
    if generation_method == "graph":
        graph_type = system.generation_data.get("graph_type") if system.generation_data else None
        graph_param = system.generation_data.get("param") if system.generation_data else None

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "system_id", "local_minima", "observed_optimal", "distribution",
            "param_value", "gap_to_local_min", "system_generation_type", "graph_type",
            "graph_param", "trial_num", "timestamp"
        ])

        if csvfile.tell() == 0:
            writer.writeheader()

        try:
            print(f"Writing trial log to: {os.path.abspath(output_path)}")
            writer.writerow({
                "system_id": system.id,
                "local_minima": local_min,
                "observed_optimal": score,
                "distribution": distribution,
                "param_value": param,
                "gap_to_local_min": (score - local_min) if (score is not None and local_min is not None) else None,
                "system_generation_type": generation_method,
                "graph_type": graph_type,
                "graph_param": graph_param,
                "trial_num": trial_num,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"ERROR while writing to CSV: {e}")
