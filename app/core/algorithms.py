import random
from copy import deepcopy
from itertools import product

from app.utils.common_utils import generate_animation

#TODO:: define stopping criteria for trial
# 1. system has converged (system score has not changed for the last N iterations)
# 2. agents have not change state for the last N iterations (tune N)

def best_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    """
    Randomly select agents from system, evaluate the best action for the agent.
    The best action for the agent is what is best for the overall system. Each agent will attempt to maximize
    overall system score.

    :param trial_num: current simulation trial
    :param data_collector: class to save data from simulation and export to json
    :param generate_graphics: boolean to enable graphics generation
    :param system: object containing all experiment data
    :param max_iterations: iteration count for given system, each new trial will reset this value
    :return: system score after running simulation
    """
    print(f"Beginning simulation with {max_iterations} iterations using best_response algorithm.")
    iteration = 0

    # list for rendering simulation gif files
    systems = []

    while iteration < max_iterations:
        iteration += 1
        agent = random.choice(system.agents)

        # Evaluate all possible actions
        action_scores = {
            frozenset(action): agent.evaluate_action(action, system, system.utility_function)
            for action in agent.action_set
        }

        # Choose the best action deterministically
        max_score = max(action_scores.values())
        best_actions = [a for a in action_scores if action_scores[a] == max_score]
        best_action = random.choice(best_actions)

        agent.current_action = set(best_action)

        if generate_graphics:
            systems.append(deepcopy(system))

        if data_collector is not None:
            data_collector.log(trial_num+1, iteration, deepcopy(system), "best_response")
        else:
            print("No reference to data collector passed, data will not be saved.")

        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

    if generate_graphics:
        generate_animation(systems=systems)

    return system.system_score()

def approximate_best_response(system=None, max_iterations=50, beta=0.5, generate_graphics=False, data_collector=None, trial_num=0):
    """
    Randomly select agents from system, evaluate all action scores for the agent. Scale each action
    based on passed beta value. Create a probability distribution using these scaled values. Select the
    most likely action from the probability distribution.

    :param trial_num: current simulation trial
    :param data_collector: class to save data from simulation and export to json
    :param generate_graphics: boolean to enable graphics generation
    :param beta: passed in value to weight resource values
                 if beta == 0: random choice, if beta == 1: best_response
    :param system: object containing all experiment data
    :param max_iterations: iteration count for given system, each new trial will reset this value
    :return: system score after running simulation
    """
    print(f"Beginning simulation with {max_iterations} iterations using approximate_best_response algorithm.")
    iteration = 0

    # list for rendering simulation gif files
    systems = []

    while iteration < max_iterations:
        iteration += 1
        agent = random.choice(system.agents)

        # Evaluate all possible actions
        action_scores = {
            frozenset(action): agent.evaluate_action(action, system, system.utility_function)
            for action in agent.action_set
        }

        max_score = max(action_scores.values())
        min_score = min(action_scores.values())

        # Create dict of [action -> new scaled score] based on passed beta value
        scaled_scores = {
            action: beta * (score - min_score) + (1 - beta) * (max_score - min_score)
            for action, score in action_scores.items()
        }

        # Select an action probabilistically based on scaled scores
        # TODO:: add ability to create different probability distributions
        total_scaled_score = sum(scaled_scores.values())
        if total_scaled_score > 0:
            probabilities = {action: score / total_scaled_score for action, score in scaled_scores.items()}
            selected_action = random.choices(
                population=list(probabilities.keys()),
                weights=list(probabilities.values()),
                k=1
            )[0]

            agent.current_action = set(selected_action)
        else:
            # TODO: Could add design here, maybe select highest value resource in list so far, or the one that can be covered by the most other agents
            # TODO: zeros tie break, implement some tie breaking rules
            agent.current_action = random.choice(list(agent.action_set))

        if generate_graphics:
            systems.append(deepcopy(system))

        if data_collector is not None:
            data_collector.log(trial_num+1, iteration, deepcopy(system), "approximate_best_response")
        else:
            print("No reference to data collector passed, data will not be saved.")

        if iteration % 1 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

    if generate_graphics:
        generate_animation(systems=systems)

    return system.system_score()

def ilp_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    return 0

def logit_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    return 0

def particle_swarm_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    return 0

def ant_colony_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    return 0

def brute_force(system=None):
    """
    Compute the higest attainable score from a given system configuration.

    :param system: object containing all experiment data
    :return: system score after brute force calculation
    """
    original_sys = system
    agents = system.agents

    # Extract all possible actions for each agent
    all_agent_action_sets = [agent.action_set for agent in agents]

    best_score = float('-inf')
    best_actions = None

    # iterate over all combinations of agent actions and find best score
    for actions_combination in product(*all_agent_action_sets):
        for agent, action in zip(system.agents, actions_combination):
            agent.current_action = action

        score = system.system_score()
        if score > best_score:
            best_score = score
            best_actions = system.resource_coverage

    # reset agent allocations
    for agent in system.agents:
        agent.current_action = set()

    system = original_sys

    return best_score, best_actions

# function map indexed by passed algorithm in application.properties
function_map = {
    "best_response": best_response,
    "approximate_best_response": approximate_best_response,
    "ilp_response": ilp_response,
    "logit_response": logit_response,
    "particle_swarm_response": particle_swarm_response
}