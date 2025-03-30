from itertools import product

from pulp import LpMaximize, LpProblem, LpVariable, lpSum


# TODO:: this function is broken, results do not equal brute force results
def ilp_response(system=None):
    print(f"Calculating maximum attainable system score using ilp_response algorithm.")

    agents = system.agents
    num_resources = len(system.resources)

    # Compute weight and coverage values for given system
    resource_weights = [resource.value for resource in system.resources]

    # Strip resource value from list and sort resources by id
    for id in agents:
        agents[id] = [[res_tuple[0] for res_tuple in action] for action in agents[id]]
        for idx, action in enumerate(agents[id]):
            action.sort()

    model = LpProblem("Find Optimal Allocation", LpMaximize)

    # Decision variables
    agent_selected = {i: LpVariable(f"a_{i}", cat="Binary") for i in agents}
    action_selected = {(i, k): LpVariable(f"s_{i}_{k}", cat="Binary") for i in agents for k in range(len(agents[i]))}
    resource_selected = {(i, j): LpVariable(f"x_{i}_{j}", cat="Binary") for i in agents for j in range(num_resources)}
    resource_covered = {j: LpVariable(f"t_{j}", cat="Binary") for j in range(num_resources)}

    # -------- CONSTRAINTS --------

    # To get optimal score every agent must choose an action
    for i in agents:
        model.addConstraint(agent_selected[i] == 1)

    # Each agent must select one action from its action set
    for i in agents:
        model.addConstraint(lpSum(action_selected[i, k] for k in range(len(agents[i]))) == 1)

    # Cover resources based on selected actions
    for i in agents:
        for k, resource_set in enumerate(agents[i]):
            for j in resource_set:
                model.addConstraint(resource_selected[i, j] == action_selected[i, k])

    # Ensure max_cover condition is satisfied, upper bind to ensure no over coverage
    for j in range(num_resources):
        model.addConstraint(lpSum(resource_selected[i, j] for i in agents) >= system.M * resource_covered[j])
        model.addConstraint(lpSum(resource_selected[i, j] for i in agents) <= (system.M + 1) * resource_covered[j])

    # -------- CONSTRAINTS END --------

    model.setObjective(lpSum(resource_covered[j] * resource_weights[j] for j in range(num_resources)))

    model.solve()

    selected_agents = [i for i in agents if agent_selected[i].varValue and agent_selected[i].varValue > 0.5]
    selected_sets = {
        i: [k for k in range(len(agents[i])) if action_selected[i, k].varValue and action_selected[i, k].varValue > 0.5]
        for i in selected_agents
    }
    covered_resources = [j for j in range(num_resources) if
                         resource_covered[j].varValue and resource_covered[j].varValue > 0.5]

    system.resource_coverage = {resource.id: 0 for resource in system.resources}

    for cover in covered_resources:
        system.resource_coverage[cover] = system.M

    score = sum(resource.value for resource in system.resources if system.resource_coverage[resource.id] >= system.M)

    print("Selected Agents:", selected_agents)
    print("Selected Sets:", selected_sets)
    print("Covered Resources:", covered_resources)

    print("ILP SCORE: ", score)

    system.resource_coverage = {}

    print("BF SCORE: ", brute_force(system))

    return system.score


def brute_force(system=None):
    """
    Compute the highest attainable score from a given system configuration.

    :param system: object containing all experiment data
    :return: system score after brute force calculation
    """
    agents = system.agents

    # Extract all possible actions for each agent
    all_agent_action_sets = [agent.action_set for agent in agents]

    best_score = float('-inf')

    # iterate over all combinations of agent actions and find best score
    for actions_combination in product(*all_agent_action_sets):
        for agent, action in zip(system.agents, actions_combination):
            agent.current_action = action

        score = system.system_score()
        if score > best_score:
            best_score = score

    # reset agent allocations
    for agent in system.agents:
        agent.current_action = set()
    # reset coverage map before calling other algorithms
    system.resource_coverage = {resource.id: 0 for resource in system.resources}
    system.score = 0

    return best_score


function_map = {
    "ilp_response": ilp_response,
    "brute_force": brute_force
}
