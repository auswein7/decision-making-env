import random
from copy import deepcopy
from itertools import product
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

from app.utils.common_utils import generate_animation
from app.utils.common_utils import format_agent_data


def best_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0, conv_iter=0):
    """
    Randomly select agents from system, evaluate the best action for the agent.
    The best action for the agent is what is best for the overall system. Each agent will attempt to maximize
    overall system score.

    :param trial_num: current simulation trial
    :param data_collector: class to save data from simulation and export to json
    :param generate_graphics: boolean to enable graphics generation
    :param system: object containing all experiment data
    :param max_iterations: iteration count for given system, each new trial will reset this value
    :param conv_iter: how many iterations the system state must remain the same for the system to be converged
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

        if data_collector is not None:
            data_collector.log(trial_num, iteration, deepcopy(system), "best_response")
        else:
            print("No reference to data collector passed, data will not be saved.")

        if iteration % conv_iter == 0:
            past_systems = systems[-conv_iter:]
            if calculate_system_convergence(past_systems, system):
                score = system.system_score()
                print(f"Best response system converged on iteration {iteration} with a final system score of {score}.\n"
                      f"Agent actions and simulation score stable for {conv_iter+1} iterations.\n")
                return score

        systems.append(deepcopy(system))

        # Print progress every 10000 iterations
        if iteration % 10000 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

    score = system.system_score()

    if generate_graphics:
        generate_animation(systems=systems)

    return score


def approximate_best_response(system=None, max_iterations=50, beta=0.5, generate_graphics=False, data_collector=None,
                              trial_num=0, conv_iter=0):
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
    :param conv_iter: how many iterations the system state must remain the same for the system to be converged
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
            # choose to cover the resource with the highest value if all actions results in 0 score to system
            # value is the sum of all resource values in an action
            max_resource_set = max(list(agent.action_set), key=lambda s: sum(r.value for r in s))
            agent.current_action = max_resource_set

        if iteration % conv_iter == 0:
            past_systems = systems[-conv_iter:]
            if calculate_system_convergence(past_systems, system):
                score = system.system_score()
                print(f"Best response system converged on iteration {iteration} with a final system score of {score}.\n"
                      f"Agent actions and simulation score stable for {conv_iter+1} iterations.\n")
                return score

        systems.append(deepcopy(system))

        if data_collector is not None:
            data_collector.log(trial_num, iteration, deepcopy(system), "approximate_best_response")
        else:
            print("No reference to data collector passed, data will not be saved.")

        # Log progress every 10000 iterations
        if iteration % 10000 == 0:
            print(f"Iteration {iteration}: System Score = {system.system_score()}")

    if generate_graphics:
        generate_animation(systems=systems)

    score = system.system_score()

    return score


def logit_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):

    return 0


def particle_swarm_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    return 0


def ant_colony_response(system=None, max_iterations=50, generate_graphics=False, data_collector=None, trial_num=0):
    return 0


#TODO:: this function is broken, results do not equal brute force results
def ilp_response(system=None):
    print(f"Calculating maximum attainable system score using ilp_response algorithm.")

    agents = system.agents
    num_resources = len(system.resources)

    # Compute weight and coverage values for given system
    resource_weights = [resource.value for resource in system.resources]

    agents = format_agent_data(agents)

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
    covered_resources = [j for j in range(num_resources) if resource_covered[j].varValue and resource_covered[j].varValue > 0.5]

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

    return system.system_score()


def brute_force(system=None):
    """
    Compute the highest attainable score from a given system configuration.

    :param system: object containing all experiment data
    :return: system score after brute force calculation
             coverage that attained the greatest system score
    """
    sys_copy = deepcopy(system)
    agents = sys_copy.agents

    # Extract all possible actions for each agent
    all_agent_action_sets = [agent.action_set for agent in agents]

    best_score = float('-inf')
    best_coverage = None

    # iterate over all combinations of agent actions and find best score
    for actions_combination in product(*all_agent_action_sets):
        for agent, action in zip(system.agents, actions_combination):
            agent.current_action = action

        score = system.system_score()
        if score > best_score:
            best_score = score
            best_coverage = system.resource_coverage

    # reset agent allocations
    for agent in system.agents:
        agent.current_action = set()

    return best_score, best_coverage

def calculate_system_convergence(sys_history, curr_sys):
    score_sim_count = 0
    score = curr_sys.system_score()
    score_history = [system.system_score() for system in list(reversed(sys_history))]
    # has the system score converged
    for idx, prev_score in enumerate(score_history):
        # if the score is still improving do not terminate
        if prev_score < score:
            return False
        else:
            score_sim_count+=1
        score = prev_score

    if score_sim_count >= len(sys_history):
        return True
    return False


# function map indexed by passed algorithm in application.properties
function_map = {
    "best_response": best_response,
    "approximate_best_response": approximate_best_response,
    "ilp_response":ilp_response,
    "logit_response": logit_response,
    "particle_swarm_response": particle_swarm_response,
    "ant_colony_response": ant_colony_response
}
