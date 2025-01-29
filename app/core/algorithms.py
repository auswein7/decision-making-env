import os
import random
import imageio. v2 as imageio
from itertools import product

from app.utils.common_utils import save_iteration_frame

def best_response(system, max_iterations=1000, output_gif="simulation.gif"):
    """
    Randomly select agents from system, evaluate the best action for the agent.
    The best action for the agent is what is best for the overall system. Each agent will attempt to maximize
    overall system score.

    :param system: object containing all experiment data
    :param max_iterations: iteration count for given system, each new trial will reset this value
    :param output_gif: output file name
    :return: system score after running simulation
    """
    print(f"Beginning simulation with {max_iterations} iterations using best_response algorithm.")

    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        agent = random.choice(system.agents)

        # Evaluate all possible actions
        action_scores = {
            frozenset(action): agent.evaluate_action(action, system, system.utility_function)
            for action in agent.action_set
        }

        # Choose the best action deterministically
        best_action = max(action_scores, key=lambda a: action_scores[a])
        agent.current_action = set(best_action)

        save_iteration_frame(system, iteration, frames_dir)

        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
    images = [imageio.imread(frame) for frame in frame_files]
    imageio.mimsave(output_gif, images, duration=0.3)

    print(f"GIF saved as {output_gif}")

    for frame in frame_files:
        os.remove(frame)

    os.removedirs(frames_dir)

    return system.system_score()

def approximate_best_response(system, max_iterations=1000, beta=0.5, output_gif="simulation.gif"):
    """
    Randomly select agents from system, evaluate all action scores for the agent. Scale each action
    based on passed beta value. Create a probability distribution using these scaled values. Select the
    most likely action from the probability distribution.

    :param beta: passed in value to weight resource values
                 if beta == 0: random choice, if beta == 1: best_response
    :param system: object containing all experiment data
    :param max_iterations: iteration count for given system, each new trial will reset this value
    :param output_gif: output file name
    :return: system score after running simulation
    """
    print(f"Beginning simulation with {max_iterations} iterations using approximate_best_response algorithm.")

    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    iteration = 0

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
            agent.current_score = random.choice(list(agent.action_set))

        save_iteration_frame(system, iteration, frames_dir)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

    # Create GIF from saved frames
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
    images = [imageio.imread(frame) for frame in frame_files]
    imageio.mimsave(output_gif, images, duration=0.3)

    print(f"GIF saved as {output_gif}")

    for frame in frame_files:
        os.remove(frame)

    os.removedirs(frames_dir)

    return system.system_score()


def ilp_response(system, max_iterations=1000):
    return 0

def brute_force(system):
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

    return best_score, best_actions

# function map indexed by passed algorithm in application.properties
function_map = {
    "best_response": best_response,
    "approximate_best_response": approximate_best_response,
    "ilp_response": ilp_response,
}