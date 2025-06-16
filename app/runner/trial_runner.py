import random
from copy import deepcopy

from app.core.agent_utilities import marginal_contribution_utility, equal_share_utility, optimistic_utility
from app.core.distributions import Distribution
from app.utils.constants import *

UTILITY_FUNCS = {
    MC_UTILITY: marginal_contribution_utility,
    ES_UTILITY: equal_share_utility,
    OPTIMISTIC_UTILITY: optimistic_utility
}

def run_trial(system=None, distribution="", agent_util=None, max_iterations=50, beta=0.5, temperature=1,
              data_collector=None, trial_num=0,
              conv_iter=float("inf"), data_key="", computing_minima=False):
    from app.utils.common_utils import calculate_system_convergence, log_trial_to_dataset
    """Conduct a single trial using the given system, distribution, and utility function."""
    func_args = {k: v for k, v in locals().items() if k not in EXCLUDE_KEYS}
    print(
        f"\nBeginning trial {trial_num} with {max_iterations} iterations:\n"
        f"\tdistribution: {distribution}\n"
        f"\tutility: {agent_util}\n"
        f"\tbeta: {beta}\n"
        f"\ttemperature: {temperature}\n"
        f"\tconv_iter: {conv_iter}.\n"
    )

    iteration = 0
    if data_collector is not None:
        data_collector.log(trial_num, iteration, deepcopy(system), data_key)

    prob_dist = Distribution(distribution=distribution, beta=beta, temperature=temperature)
    for agent in system.agents:
        agent.utility_function = UTILITY_FUNCS[agent_util]
        agent.current_action = set()

    system_scores = []
    while iteration < max_iterations:
        iteration += 1

        agent = random.choice(system.agents)

        candidates = list(agent.action_set) + [agent.current_action]
        action_scores = {
            frozenset(action): agent.evaluate_action(action, system, agent.utility_function)
            for action in candidates
        }

        best_action, _ = prob_dist.get_distribution()(action_scores)
        agent.current_action = set(best_action)

        system.system_score()

        if data_collector is not None:
            data_collector.log(trial_num, iteration, deepcopy(system), data_key)

        if iteration % conv_iter == 0:
            if calculate_system_convergence(system_scores, system):
                print(
                    f"\n{distribution} system converged on iteration {iteration} with a final system score of {system.score}.\n"
                    f"Simulation score stable for {conv_iter} iterations.\n")
                return system.score, func_args

        system_scores.append(system.score)

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: System Score = {system.score}")

    if system_scores:
        k = max(1, int(len(system_scores) * 0.2))
        score = sum(system_scores[-k:]) / k
    else:
        score = float('nan')

    if not computing_minima:
        log_trial_to_dataset(system, score, distribution, beta, temperature, trial_num)

    return score, func_args
