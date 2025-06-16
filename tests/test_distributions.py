import uuid
import os
import matplotlib.pyplot as plt
import numpy as np

from app.core.agent_utilities import marginal_contribution_utility, equal_share_utility, optimistic_utility
from app.core.distributions import Distribution
from app.models.agent import Agent
from app.models.system import System
from app.models.resource import Resource

from app.utils.constants import *

UTILITY_FUNCS = {
    MC_UTILITY: marginal_contribution_utility,
    ES_UTILITY: equal_share_utility,
    OPTIMISTIC_UTILITY: optimistic_utility
}

def generate_sample_system():
    resources = [
        Resource(1, 1),
        Resource(2, 2),
        Resource(3, 3),
        Resource(4, 4),
        Resource(5, 5),
    ]

    agents = [
        Agent(0, [
            {resources[0]},
            {resources[1]},
            {resources[2]},
            {resources[3]},
            {resources[4]},
        ], None),
    ]

    m = 1
    sys_id = uuid.uuid4().hex
    system = System(resources, agents, m, sys_id)
    return system


def plot_action_probabilities_over_param(
        action_scores,
        param_values,
        get_prob_dist_fn,
        title,
        x_label,
        output_filename):
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)

    all_actions = list(action_scores.keys())
    action_labels = [
        ",".join(str(r.id) for r in sorted(action))
        for action in all_actions
    ]

    action_prob_over_params = {label: [] for label in action_labels}

    for param in param_values:
        probs = get_prob_dist_fn(param)
        for label, action in zip(action_labels, all_actions):
            prob = probs.get(action, 0.0)
            action_prob_over_params[label].append(prob)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(action_labels)))
    markers = ['o', 's', '^', 'D', 'x', '*', 'v']

    for idx, (label, prob_list) in enumerate(action_prob_over_params.items()):
        jitter = np.array(prob_list) + (idx - len(action_labels) / 2) * 0.005
        plt.plot(param_values, jitter,
                 marker=markers[idx % len(markers)],
                 linestyle='-',
                 label=label,
                 color=colors[idx])

    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Selection Probability", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(param_values)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, output_filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def test_approximate_best_response():
    system = generate_sample_system()
    beta_values = [1.0, 0.8, 0.5, 0.3, 0.0]

    for agent in system.agents:
        agent.utility_function = UTILITY_FUNCS[ES_UTILITY]
        agent.current_action = set()

    agent = system.agents[0]
    candidates = list(agent.action_set)
    action_scores = {
        frozenset(action): agent.evaluate_action(action, system, agent.utility_function)
        for action in candidates
    }

    def get_probs_for_beta(beta):
        dist = Distribution(distribution=APPROX_BEST_RESPONSE, beta=beta, temperature=0.0)
        _, probs = dist.get_distribution()(action_scores)
        return probs or {}

    plot_action_probabilities_over_param(
        action_scores,
        param_values=beta_values,
        get_prob_dist_fn=get_probs_for_beta,
        title="ABR Probs vs Beta",
        x_label="beta",
        output_filename="abr_prob_curve.png"
    )


def test_logit_response():
    system = generate_sample_system()
    temperature_values = [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

    for agent in system.agents:
        agent.utility_function = UTILITY_FUNCS[ES_UTILITY]
        agent.current_action = set()

    agent = system.agents[0]
    candidates = list(agent.action_set)
    action_scores = {
        frozenset(action): agent.evaluate_action(action, system, agent.utility_function)
        for action in candidates
    }

    def get_probs_for_temp(temp):
        dist = Distribution(distribution=LOGIT_RESPONSE, beta=0.0, temperature=temp)
        _, probs = dist.get_distribution()(action_scores)
        return probs or {}

    plot_action_probabilities_over_param(
        action_scores,
        param_values=temperature_values,
        get_prob_dist_fn=get_probs_for_temp,
        title="Log Linear Learning Probs vs Temp",
        x_label="temp",
        output_filename="logit_prob_curve.png"
    )
