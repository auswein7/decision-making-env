import random

def best_response(system, max_iterations=1000):
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

        # Choose the action with the highest payoff (breaking ties randomly)
        best_action = max(action_scores, key=lambda a: action_scores[a])
        agent.current_action = set(best_action)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")