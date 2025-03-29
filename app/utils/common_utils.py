import json
import math
import os
import time
import uuid
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from app.models.agent import Agent
from app.models.resource import Resource
from app.core.algorithms import function_map
from app.models.system import System
from app.utils.constants import *

# TODO:: COMMENT FUNCS IN THIS FILE

# TODO:: REFACTOR TO COMPLY WITH NEW CHANGES TO CLASSES
def load_scenario_from_json(file_path):
    """
    Load scenario data from the scenarios directory

    :param file_path: path to scenario json file
    :return: sys, algorithm
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

        resources = [
            Resource(r["id"], r["value"])
            for r in data["resources"]
        ]

        agents = [
            Agent(a["id"], (a["action_set"]))
            for a in data["agents"]
        ]

        # convert agent action set to frozenset of resources
        for agent in agents:
            for i, subset in enumerate(agent.action_set):
                agent.action_set[i] = [r for r in resources if r.id in subset]
            agent.action_set = {frozenset(action) for action in agent.action_set}

        m = data["system"]["m"]
        sys = System(resources=resources, agents=agents, m=m)
        return sys


def export_scenario_to_json(system=None, file_path="app/out"):
    """
    Export a given random system to json file to be reloaded in future runner.

    :param system: experiment system configuration
    :param file_path: path to scenario json file
    :return: filename: name of created scenario json file
    """
    uuid_str = uuid.uuid4()

    # visualize a system per configuration saved
    visualize_system_configuration(system, uuid_str, file_path)

    file_name = os.path.join(file_path, "saved_models", f"{uuid_str}.json")
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
        "m": system.M,
    }

    simulation_data = {
        "resources": resources_data,
        "agents": agents_data,
        "system": system_data,
    }

    with open(file_name, 'w') as json_file:
        json.dump(simulation_data, json_file, indent=4)

    return file_name


# TODO:: REFACTOR, FAR TOO BUSY WITH MANY RESOURCES AND MANY AGENTS
def visualize_system_configuration(system=None, uuid_str=None, file_path="app/out"):
    """
    Given a system configuration, render a graph representing the coverage of all agents in
    the system.

    :param file_path: save directory of this visualization
    :param uuid_str: UUID of json file that will be exported in saved_models
    :param system: initial system state in simulation
    :return:
    """
    file_name = os.path.join(file_path, "saved_models", f"{uuid_str}.png")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    g = nx.DiGraph()

    agent_color = "blue"
    uncovered_resource_color = "green"
    subsets = {}

    for agent in system.agents:
        node_id = f"A{agent.id}"
        g.add_node(node_id, color=agent_color, label=f"A{agent.id}")
        subsets[node_id] = 1  # agents -> subset 1

    for resource in system.resources:
        node_id = f"R{resource.id}"
        g.add_node(node_id, color=uncovered_resource_color, label=f"R{resource.id}\nVal: {resource.value}")
        subsets[node_id] = 0  # resources -> subset 2

    for agent in system.agents:
        for i, action_set in enumerate(agent.action_set):
            for resource in action_set:
                g.add_edge(f"A{agent.id}", f"R{resource.id}", label=f"{agent.id}: Action Set {i + 1}")

    nx.set_node_attributes(g, subsets, "subset")

    # alter k val to change spacing
    pos = nx.spring_layout(g, k=1.5, seed=42)

    colors = [g.nodes[n]["color"] for n in g.nodes]
    labels = {n: g.nodes[n]["label"] for n in g.nodes}
    edge_labels = {(u, v): d["label"] for u, v, d in g.edges(data=True) if "label" in d}

    plt.figure(figsize=(14, 10))
    nx.draw(g, pos, with_labels=True, labels=labels, node_color=colors, edge_color="gray", node_size=3000, font_size=14,
            font_weight="bold")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=10)

    plt.title("System Configuration")
    plt.savefig(file_name)


def generate_param_analysis_plot(data, sys_optimal):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for (utility, var_name), values in data.items():
        folder_path = os.path.join(JSON_SAVE_PATH, f"{utility}-{var_name}-whisker")
        os.makedirs(folder_path, exist_ok=True)

        normalized_scores = [np.array(scores) / sys_optimal for scores in values['scores']]

        plt.figure(figsize=(8, 5))
        plt.boxplot(normalized_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.xticks(ticks=range(1, len(values['x']) + 1), labels=values['x'])

        plt.xlabel(f"{var_name}", fontsize=8)
        plt.ylabel("Normalized Score", fontsize=8)
        plt.title(f"{var_name} vs System Score ({utility})", fontsize=10)
        plt.grid(alpha=0.5)

        filename = os.path.join(folder_path, f"{var_name}_{utility}_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_scores_by_rank(data, title='', x_label='', y_label='Normalized System Scores', sys_opts=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(JSON_SAVE_PATH, f"{title.replace(' ', '_')}")
    os.makedirs(folder_path, exist_ok=True)

    sorted_items = sorted(data.items(), key=lambda x: x[1][0])

    x_labels = []
    box_data = []

    for system_id, values in sorted_items:
        rank = values[0]
        scores = values[1:]

        opt = sys_opts[system_id]
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

    filename = os.path.join(folder_path, f"{title.replace(' ', '_')}_{x_label.replace(' ', '_')}_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_zoomed_analysis_plot(data, sys_optimal):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for (utility, var_name), values in data.items():
        folder_path = os.path.join(JSON_SAVE_PATH, f"{utility}-{var_name}-zoom")
        os.makedirs(folder_path, exist_ok=True)

        for param_value, scores in zip(values['x'], values['scores']):
            normalized_scores = np.array(scores) / sys_optimal

            plt.figure(figsize=(8, 5))
            plt.plot(range(len(normalized_scores)), normalized_scores, marker='o', color="blue")
            plt.xlabel('Repetition', fontsize=8)
            plt.ylabel('Norm Score', fontsize=8)
            plt.title(f'{var_name}:{param_value} with {utility}', fontsize=10)
            plt.grid(alpha=0.5)

            filename = os.path.join(folder_path, f"{var_name}_{utility}_{param_value}_{timestamp}.png")
            plt.savefig(filename, dpi=300)
            plt.close()

def plot_optimal_iterations(data, sys_optimal):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    title = "Normalized Scores vs Iterations"
    folder_path = os.path.join(JSON_SAVE_PATH, title.replace(" ", "_"))
    os.makedirs(folder_path, exist_ok=True)

    sorted_items = sorted(data.items())
    x_labels = [str(k) for k, _ in sorted_items]

    box_data = []
    for _, scores in sorted_items:
        norm_scores = [s / sys_optimal for s in scores]
        box_data.append(norm_scores)

    plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, vert=True, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
    plt.xticks(ticks=range(1, len(x_labels) + 1), labels=x_labels)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Normalized Score")
    plt.grid(alpha=0.5)
    plt.tight_layout()

    filename = os.path.join(folder_path, f"{title.replace(' ', '_')}_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

def format_agent_data(agents):
    out = {agent.id: [] for agent in agents}
    for agent in agents:
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                sub_list.append((resource.id, resource.value))
            out[agent.id].append(sub_list)
    return out


def compute_overlap_density(agents, resources):
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

    return total_overlap / count


def compute_agent_resource_entropy(agents, resources):
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
    system_difficulties_fm = {}
    system_difficulties_re = {}
    system_difficulties_od = {}

    for system in systems:
        system_difficulties_fm[system.id] = [compute_feasibility_margin(system.agents, system.resources, system.M)]
        system_difficulties_re[system.id] = [compute_agent_resource_entropy(system.agents, system.resources)]
        system_difficulties_od[system.id] = [compute_overlap_density(system.agents, system.resources)]

    system_difficulties_fm = dict(sorted(system_difficulties_fm.items(), key=lambda x: x[1], reverse=True))
    system_difficulties_re = dict(sorted(system_difficulties_re.items(), key=lambda x: x[1], reverse=True))
    system_difficulties_od = dict(sorted(system_difficulties_od.items(), key=lambda x: x[1]))

    return system_difficulties_fm, system_difficulties_re, system_difficulties_od

def get_iteration_range(initial_iters, relative_step):
    step = initial_iters * relative_step
    left = [int(initial_iters - step * i) for i in reversed(range(5))]  # center and 4 left
    right = [int(initial_iters + step * i) for i in range(1, 6)]        # 5 right
    return left + right

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


def calc_and_time_optimal(system):
    print("Calculating system optimal score")
    start_t = time.time()
    optimal_score = function_map.get(BRUTE_FORCE)(system)
    end_t = time.time()
    print(
        f"Optimal System score for this configuration {optimal_score:.3f}, calculated in {end_t - start_t:.3f} seconds")
    return optimal_score

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
