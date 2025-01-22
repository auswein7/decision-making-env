import random

def best_response(system, max_iterations=1000):
    """
    Randomly select agents from system, evaluate the best action for the agent.
    The best action for the agent is what is best for the overall system. Each agent will attempt to maximize
    overall system score.

    :param system: object containing all experiment data
    :param max_iterations: iteration count for given system, each new trial will reset this value
    :return: none
    """
    print(f"Beginning simulation with {max_iterations} iterations using best_response algorithm.")
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Select agent randomly
        agent = random.choice(system.agents)

        # Evaluate all possible actions
        action_scores = {
            frozenset(action): agent.evaluate_action(action, system, system.utility_function)
            for action in agent.action_set
        }

        # Choose the action with the highest payoff
        best_action = max(action_scores, key=lambda a: action_scores[a])
        agent.current_action = set(best_action)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

def approximate_best_response(system, max_iterations=1000):
    return 0

def ilp_response(system, max_iterations=1000):
    return 0