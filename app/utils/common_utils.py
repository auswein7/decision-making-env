import uuid
import json
import os
import io

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from app.core.system import System
from app.core.utils import system_utility
from app.models.resource import Resource
from app.models.agent import Agent

from app.utils.constants import JSON_SAVE_PATH


def load_scenario_from_json(file_path):
    """
    Load scenario data from the scenarios directory

    :param file_path: path to scenario json file
    :return: sys, algorithm
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

        algorithm = data["algorithm"]

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
        sys = System(resources=resources, agents=agents, m=m, utility_function=system_utility)
        return sys, algorithm


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
        "utility_function": system.utility_function.__name__
    }

    simulation_data = {
        "resources": resources_data,
        "agents": agents_data,
        "system": system_data,
    }

    with open(file_name, 'w') as json_file:
        json.dump(simulation_data, json_file, indent=4)

    return file_name


# TODO:: if I hover an agent, highlight the resources that it covers
# TODO:: put agents on one side of render, resources on the other
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


def generate_animation(systems=None, output_gif="simulation.gif"):
    """
    Save the agents' current action choice and render matplotlib plt.

    :param systems: History of system objects throughout simulation
    :param output_gif: Name of simulation gif file
    :return: None
    """
    print("Rendering simulation animation!")
    images = []

    for iteration, system in enumerate(systems):
        fig, ax = plt.subplots(figsize=(8, 6))

        covered_resources = {id for id in system.resource_coverage if system.resource_coverage[id] >= system.M}

        # Plot resources as squares
        for resource in system.resources:
            x, y = resource.id * 2, 0
            color = "green" if resource.id in covered_resources else "red"
            ax.scatter(x, y, s=500, c=color, marker="s", label="Resource")
            ax.text(x, y, f"{resource.id}\n({resource.value})", fontsize=10, ha="center", va="center", color="white")

        # Plot agents as circles
        for idx, agent in enumerate(system.agents):
            x, y = idx * 2, 3
            ax.scatter(x, y, s=500, c="blue", marker="o", label="Agent")
            ax.text(x, y, str(agent.id), fontsize=10, ha="center", va="center", color="white")

            # Draw connections between agent and selected resources
            if agent.current_action:
                for resource in agent.current_action:
                    ax.plot([x, resource.id * 2], [y, 0], 'k-', alpha=0.5)

        # Add iteration and system score
        ax.set_xlim(-2, len(system.resources) * 2 + 2)
        ax.set_ylim(-1, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Iteration {iteration}: System Score = {system.system_score()}")

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).copy()
        images.append(img)

        buf.close()
        plt.close(fig)

    images[0].save(fp=output_gif, save_all=True, append_images=images[1:], duration=180, loop=0)

    print(f"GIF saved as {output_gif}")

def generate_param_analysis_plot(x, y, var_name):
    filename = os.path.join(JSON_SAVE_PATH,f"beta_vs_sys_score.png")
    os.makedirs(JSON_SAVE_PATH, exist_ok=True)

    plt.figure()
    plt.ylabel("System Score per trial")
    plt.xlabel(f"{var_name}")
    plt.title(F"{var_name} vs System Score")
    plt.plot(x, y)

    plt.savefig(filename)


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
