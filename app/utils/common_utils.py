import json
import math
import os
import time
import random
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
import colorsys

from app.core.algorithms import function_map
from app.models.agent import Agent
from app.models.resource import Resource
from app.models.system import System
from app.utils.constants import *

loaded_systems = []


def load_scenario_from_json(system_file_uuids):
    """
    Load scenario data from the scenarios directory

    :param system_file_uuids: uuids of target systems to load
    :return: systems: the target systems requested
    """
    systems = []
    for sys_id in system_file_uuids:
        if sys_id not in loaded_systems:
            loaded_systems.append(sys_id)

            file_path = os.path.join(JSON_LOAD_PATH, sys_id + ".json")
            with open(file_path, 'r') as file:
                data = json.load(file)

                resources = [
                    Resource(r["id"], r["value"])
                    for r in data["resources"]
                ]

                agents = [
                    Agent(a["id"], (a["action_set"]), None)
                    for a in data["agents"]
                ]

                # convert agent action set to frozenset of resources
                for agent in agents:
                    for i, subset in enumerate(agent.action_set):
                        agent.action_set[i] = {r for r in resources if r.id in subset}

                m = data["system"]["m"]
                id = data["system"]["id"]
                sys = System(resources=resources, agents=agents, m=m, id=id)
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
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
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

        normalized_scores = [np.array(scores) / sys_optimal for scores in values['scores']]

        plt.figure(figsize=(8, 5))
        plt.boxplot(normalized_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.xticks(ticks=range(1, len(values['x']) + 1), labels=values['x'])

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
            normalized_scores = np.array(scores) / sys_optimal

            plt.figure(figsize=(8, 5))
            plt.ylim(0, 1)
            plt.plot(range(len(normalized_scores)), normalized_scores, marker='o', color="blue")
            plt.xlabel('Repetition', fontsize=8)
            plt.ylabel('Norm Score', fontsize=8)
            plt.title(f'{var_name}:{param_value} with {utility}', fontsize=10)
            plt.grid(alpha=0.5)

            filename = os.path.join(folder_path, f"{sys_id}_{param_value}_{timestamp}.png")
            plt.savefig(filename, dpi=300)
            plt.close()


def plot_optimal_iterations(data, sys_optimal, sys_id):
    """
    For all distribution iterations render the box plot score of the repetitions.

    :param data: formatted dictionary of run data
    :param sys_optimal: optimal score for the tested system
    :param sys_id: system.id uuid
    :return: None
    """
    # TODO:: add {sys_uuid}_{iterations}_{distribution}_{utility} for title as well as save file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    title = "Normalized Scores vs Iterations"
    folder_path = os.path.join(JSON_SAVE_PATH, title.replace(" ", "_").lower())
    os.makedirs(folder_path, exist_ok=True)

    sorted_items = sorted(data.items())
    x_labels = [str(k) for k, _ in sorted_items]

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

    return total_overlap / count


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
    Compute difficulties for system across all three metrics.

    :param systems: created systems for system analysis run
    :return: three dictionaries of system difficulties
    """
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
    """
    Format data for parameter analysis run plotting.

    :param score_history: full history over repetitions in a run.
    :return: formateed plotting data
    """
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


def calc_and_time_optimal(system, init_from_opt):
    """
    Calculate the system optimal score and output time taken to calculate.

    :param system: system to compute optimal score for.
    :param init_from_opt: clear the agent actions or not from brute force calc
    :return: optimal score of the system
    """
    print("Calculating system optimal score")
    start_t = time.time()
    optimal_score, optimal_coverage = function_map.get(BRUTE_FORCE)(system, init_from_opt)
    end_t = time.time()
    print(
        f"Optimal System score for this configuration {optimal_score:.3f}, calculated in {end_t - start_t:.3f} seconds")
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
