from itertools import product

def brute_force(system=None, init_from_opt=False, cutoff=500000):
    """
    Compute the highest attainable score from a given system configuration.

    :param cutoff: when to not calculate the highest attainable score
    :param system: object containing all experiment data
    :param init_from_opt: do not reset agents actions after optimal calc
    :return: system score after brute force calculation
    """
    print(f"[brute_force] Calculating for agent action combinations of {system.action_combinations}")
    if system.action_combinations.get(system.id)[0] is not None and system.action_combinations.get(system.id)[0]  > cutoff:
        print(f"[brute_force] Skipping brute force: {system.action_combinations} combinations exceeds cutoff of {cutoff}")
        return -1, {}

    agents = system.agents

    # Extract all possible actions for each agent
    all_agent_action_sets = [agent.action_set for agent in agents if len(agent.action_set) > 0]

    best_score = float('-inf')

    # iterate over all combinations of agent actions and find best score
    for actions_combination in product(*all_agent_action_sets):
        for agent, action in zip(system.agents, actions_combination):
            agent.current_action = action

        score = system.system_score()
        if score > best_score:
            best_score = score

    # reset agent allocations
    if not init_from_opt:
        for agent in system.agents:
            agent.current_action = set()

    optimal_coverage = system.resource_coverage
    # reset coverage map before calling other algorithms
    if not init_from_opt:
        system.resource_coverage = {resource.id: 0 for resource in system.resources}

    system.score = 0

    return best_score, optimal_coverage


function_map = {
    "brute_force": brute_force
}
