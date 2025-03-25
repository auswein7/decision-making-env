import json
import os
import uuid

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from app.core.system import System
from app.models.agent import Agent
from app.models.resource import Resource
from app.utils.constants import *


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


def generate_param_analysis_plot(data, sys_optimal):
    for (utility, var_name), values in data.items():
        folder_path = os.path.join(JSON_SAVE_PATH, f"{utility}-{var_name}-whisker")
        os.makedirs(folder_path, exist_ok=True)

        normalized_scores = [np.array(scores) / sys_optimal if sys_optimal != 0 else np.zeros_like(scores) for scores in
                             values['scores']]

        plt.figure(figsize=(8, 5))
        plt.boxplot(normalized_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.xticks(ticks=range(1, len(values['x']) + 1), labels=values['x'])

        plt.xlabel(f"{var_name}", fontsize=8)
        plt.ylabel("Normalized Score", fontsize=8)
        plt.title(f"{var_name} vs System Score ({utility})", fontsize=10)
        plt.grid(alpha=0.5)

        filename = os.path.join(folder_path, f"{var_name}_{utility}.png")
        plt.savefig(filename, dpi=300)
        plt.close()


def generate_zoomed_analysis_plot(data, sys_optimal):
    for (utility, var_name), values in data.items():
        folder_path = os.path.join(JSON_SAVE_PATH, f"{utility}-{var_name}-zoom")
        os.makedirs(folder_path, exist_ok=True)

        for param_value, scores in zip(values['x'], values['scores']):
            normalized_scores = np.array(scores) / sys_optimal if sys_optimal != 0 else np.zeros_like(scores)

            plt.figure(figsize=(8, 5))
            plt.plot(range(len(normalized_scores)), normalized_scores, marker='o', color="blue")
            plt.xlabel('Repetition', fontsize=8)
            plt.ylabel('Norm Score', fontsize=8)
            plt.title(f'{var_name}:{param_value} with {utility}', fontsize=10)
            plt.grid(alpha=0.5)

            filename = os.path.join(folder_path, f"{var_name}_{utility}_{param_value}.png")
            plt.savefig(filename, dpi=300)
            plt.close()


def format_agent_data(agents):
    out = {agent.id: [] for agent in agents}
    for agent in agents:
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                output_tuple = (resource.id, resource.value)
                sub_list.append(output_tuple)
            out[agent.id].append(sub_list)
    return out
