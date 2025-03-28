import random
from copy import deepcopy

from app.core.distributions import Distribution
from app.core.utils import marginal_contribution_utility, equal_share_utility, optimistic_utility

from app.utils.constants import *

UTILITY_FUNCS = {
    MC_UTILITY: marginal_contribution_utility,
    ES_UTILITY: equal_share_utility,
    OPTIMISTIC_UTILITY: optimistic_utility
}


def run_trial(system=None, distribution="", agent_util=None, max_iterations=50, beta=0.5, temperature=1,
              data_collector=None, trial_num=0,
              conv_iter=float("inf"), data_key=""):
    func_args = {k: v for k, v in locals().items() if k not in EXCLUDE_KEYS}
    print(
        f"\nBeginning trial {trial_num} with {max_iterations} iterations using {PROB_RESPONSE} algorithm with params:\n"
        f"beta: {beta}\n"
        f"temperature: {temperature}\n"
        f"distribution: {distribution}\n"
        f"conv_iter: {conv_iter}.\n")

    iteration = 0
    # Log initial state
    if data_collector is not None:
        data_collector.log(trial_num, iteration, deepcopy(system), data_key)

    prob_dist = Distribution(distribution=distribution, beta=beta, temperature=temperature)
    for agent in system.agents:
        agent.utility_function = UTILITY_FUNCS[agent_util]
        agent.current_action = set()

    # list for system convergence calculation
    system_scores = []
    while iteration < max_iterations:
        iteration += 1

        agent = random.choice(system.agents)

        # TODO:: Should agents be allowed to not change their action?
        # Evaluate all possible actions
        action_scores = {
            frozenset(action): agent.evaluate_action(action, system, agent.utility_function)
            for action in agent.action_set
        }

        best_action = prob_dist.get_distribution()(action_scores)

        agent.current_action = set(best_action)

        # calculate system score after agent has chosen action
        system.system_score()

        if data_collector is not None:
            data_collector.log(trial_num, iteration, deepcopy(system), data_key)

        if iteration % conv_iter == 0:
            if calculate_system_convergence(system_scores, system):
                print(
                    f"\n{distribution} system converged on iteration {iteration} with a final system score of {system.score}.\n"
                    f"Simulation score stable for {conv_iter} iterations.\n")
                return system.score

        system_scores.append(system.score)

        # Print progress every 1000 iterations
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: System Score = {system.score}")

    return sum(system_scores[-int(len(system_scores) * 0.2):]) / int(len(system_scores) * 0.2), func_args


def calculate_system_convergence(score_history, curr_sys):
    """
    Compute system convergence based on stability of system score.

    :param score_history: Conv_iter previous system scores
    :param curr_sys: current system at this iteration
    :return: true if system has converged, false otherwise.
    """
    score_sim_count = 0
    score = curr_sys.score
    # has the system score converged
    for idx, prev_score in enumerate(reversed(score_history)):
        # if the score is still improving do not terminate
        if prev_score < score:
            return False
        else:
            score_sim_count += 1
        score = prev_score

    if score_sim_count >= len(score_history):
        return True
    return False
