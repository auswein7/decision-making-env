import json
import os
import uuid

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from app.core.system import System
from app.core.utils import global_visibility_utility
from app.models.agent import Agent
from app.models.resource import Resource
from app.utils.constants import JSON_SAVE_PATH


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
        sys = System(resources=resources, agents=agents, m=m, utility_function=global_visibility_utility)
        return sys


def export_scenario_to_json(system=None, file_path="app/out"):
    """
    Export a given random system to json file to be reloaded in future experiments.

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


def generate_param_analysis_plot(x, y, var_name, sys_optimal):
    """
    Generate plot given system score per trial, targeting a specific parameter.



    :param x: axis data, parameter value for trial
    :param y: axis data, system score for trial
    :param var_name: name of the target parameter
    :param sys_optimal: optimal score for normalization
    :return: None
    """
    filename = os.path.join(JSON_SAVE_PATH, f"{var_name}_vs_sys_score.png")
    os.makedirs(JSON_SAVE_PATH, exist_ok=True)


    normalized_data = [np.array(score) / sys_optimal for score in y]

    plt.figure(figsize=(8, 5))
    plt.plot(x, normalized_data, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel(f"{var_name}", fontsize=12)
    plt.ylabel("System Score per trial", fontsize=12)
    plt.title(f"{var_name} vs System Score", fontsize=14, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def generate_histogram_analysis_plot(data, var_name, bins, sys_optimal):
    """
    Generate individual histograms for each param val and save separately to same dir.

    :param sys_optimal: highest possible score given sys config. Used for norm [0, optimal]
    :param bins: Histogram bin count
    :param data: Dict of param val -> scores over trial_repetitions
    :param var_name: Variable name used for file naming and titles
    """
    folder_path = os.path.join(JSON_SAVE_PATH, f"{var_name}_histograms")
    os.makedirs(folder_path, exist_ok=True)

    normalized_data = {
        param: np.array(scores) / sys_optimal if sys_optimal != 0 else np.zeros_like(scores)
        for param, scores in data.items() if param[0] == var_name[0]
    }

    bins = np.linspace(0, 1, bins)

    for (param, scores) in normalized_data.items():
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=bins, alpha=0.7, color="blue", edgecolor='black')

        plt.xlabel('Normalized System Score', fontsize=10)
        plt.ylabel('Occurrences', fontsize=10)
        plt.title(f'Histogram for {var_name} {param}', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        filename = os.path.join(folder_path, f"{var_name}_{param}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

def format_agent_data(agents):
    """
    Extract nested set data to export agent data to json files.

    :param agents: list of agents
    :return: formatted dictionary of agents -> resources
    """
    out = {agent.id: [] for agent in agents}
    for agent in agents:
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                output_tuple = (resource.id, resource.value)
                sub_list.append(output_tuple)
            out[agent.id].append(sub_list)
    return out
