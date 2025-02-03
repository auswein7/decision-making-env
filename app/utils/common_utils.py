import json
import uuid
import os
import io
import matplotlib.pyplot as plt
from PIL import Image

from app.core.system import System
from app.core.utils import system_utility
from app.models.resource import Resource
from app.models.agent import Agent

from app.utils.logger import Logger

logger = Logger.get_logger()

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

def export_scenario_to_json(system=None, algorithm="", file_path="app/out"):
    """
    Export a given random system to json file to be reloaded in future experiments.

    :param system: experiment system configuration
    :param algorithm: Chosen algorithm to run experiment
    :param file_path: path to scenario json file
    :return: None
    """
    filename = os.path.join(file_path, f"{uuid.uuid4()}.json")

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
        "algorithm": algorithm
    }

    with open(filename, 'w') as json_file:
        json.dump(simulation_data, json_file, indent=4)

    logger.info(f"Simulation results saved to {filename}")

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


def log_system_properties(system, trial_num):
    """
    Export information to the 'experiment.log' file

    :param system: system for the given trial
    :param trial_num: current running trial
    :return: none
    """
    logger.info(f"Trial {trial_num+1}")
    logger.info(f"Max Cover {system.M}")
    logger.info(f"Num Resources: {len(system.resources)}")

    out_list = []
    for resource in system.resources:
        out_list.append((resource.id, resource.value))
    logger.info(f"Resource List: {out_list}")

    logger.info(f"Num Agents: {len(system.agents)}")
    for agent in system.agents:
        out_list = []
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                output_tuple = (resource.id, resource.value)
                sub_list.append(output_tuple)
            out_list.append(sub_list)
        logger.info(f"Agent {agent.id} Action Set: {out_list}")

def log_agent_allocation(system):
    """
    Export information to the 'experiment.log' file. This data is collected at the end of the run
    and shows the actions the agents chose.

    :param system: system state after trial has completed
    :return: none
    """
    logger.info(f"Simulation score {system.system_score()}")
    for agent in system.agents:
        out_list = []
        for resource in agent.current_action:
            out_list.append((resource.id, resource.value))

        logger.info(f"Agent {agent.id} Covers: {out_list}")

    logger.info(f"System resource coverage: {system.resource_coverage}")