import json
import os
import uuid
from copy import deepcopy

from app.core.algorithms import brute_force
from app.utils.common_utils import format_agent_data

from app.utils.constants import JSON_SAVE_PATH


class DataCollector:
    """
    DataCollector class to track all simulation data and compile into a simulation summary json.

    Attributes:
        results: data aggregated through scenario execution.
    """

    def __init__(self, algorithms):
        self.results = {algo_name: [] for algo_name in algorithms}

    def log(self, trial, iteration, system, algorithm):
        """
        Log the current state during every iteration. Called trial * iteration times throughout a run.

        Attributes:
            trial: trial number for this simulation.
            iteration: iteration number for this trial.
            system: configuration at trial/iteration
            algorithm: algorithm used for index into results dict
        """
        self.results[algorithm].append({
            "trial": trial,
            "iteration": iteration,
            "system": system
        })

    def clear_data(self, algorithms):
        """ Reset simulation data to initialization"""
        self.results = {algo_name: [] for algo_name in algorithms}

    def format_results_data(self):
        """ format results into nested dict, algo -> trial num -> iteration data """
        formatted_results = {algo: {} for algo in self.results}
        for entry in self.results:
            for iteration_data in self.results[entry]:
                trial = iteration_data["trial"]
                if trial not in formatted_results[entry]:
                    formatted_results[entry][trial] = []
                formatted_results[entry][trial].append(iteration_data)
        return formatted_results

    def summarize_results(self, save_files_per_trial):
        """
        Compile all stored data in self.results.

        Attributes:
            save_files_per_trial: dict of trial -> system json uuid used that trial
        """
        formatted_results = self.format_results_data()

        for algo in formatted_results.keys():
            # set up output data, keyed by trial number
            unique_trials = list(range(0, len(formatted_results[algo])))
            sim_summary = {trial: {} for trial in unique_trials}

            for trial, data in formatted_results[algo].items():
                # get the system from trial 0 iteration 1
                original_system = [entry for entry in data if entry["iteration"] == 1][0].get("system")

                for agent in original_system.agents:
                    agent.current_action = set()

                # get the system from final trial final iteration
                final_system = formatted_results[algo][trial][-1]["system"]

                brute_force_score = brute_force(deepcopy(original_system))[0]
                if brute_force_score == 0:
                    print("No resources can be covered by this system")
                    return -1

                agents = final_system.agents
                sim_score = final_system.system_score()
                coverage_map = final_system.resource_coverage
                resources = final_system.resources
                formatted_resources = []
                for resource in resources:
                    formatted_resources.append((resource.id, resource.value))
                max_cover = final_system.M
                formatted_agents = format_agent_data(agents)

                # pull the final state of agent decisions
                agent_actions = {}
                for agent in agents:
                    res_list = []
                    for resource in agent.current_action:
                        res_list.append((resource.id, resource.value))
                    agent_actions[agent.id] = res_list

                resources_covered = sum(1 for resource in resources if coverage_map[resource.id] >= max_cover)
                over_coverage_map = {resource.id: coverage_map[resource.id] for resource in resources if
                                     coverage_map[resource.id] > max_cover}
                overhead_actions, net_contributions = DataCollector.calculate_overhead_net_contribution(data=data)
                best_system, best_iteration = DataCollector.get_best_system_iter(data=data)

                sim_summary[trial] = ({
                    "max_cover": max_cover,
                    "num_agents": len(agents),
                    "agent_ids": [a.id for a in agents],
                    "action_sets": formatted_agents,
                    "agent_allocations": agent_actions,
                    "num_resources": len(resources),
                    "resource_values": formatted_resources,
                    "resource_coverage": coverage_map,
                    "resource_coverage_percentage": resources_covered / len(resources),
                    "over_covered_resources": over_coverage_map,
                    "max_possible_score": brute_force_score,
                    "simulation_score": sim_score,
                    "grade": str((sim_score / brute_force_score) * 100) + "%",
                    "best_system_coverage": format_agent_data(best_system.agents),
                    "best_system_score": best_system.system_score(),
                    "iteration_of_best_system": best_iteration,
                    "agent_total_actions": DataCollector.count_agent_actions(data=data),
                    "resource_popularity": DataCollector.calculate_resource_popularity(data=data),
                    "agent_contributions": DataCollector.calculate_agent_contribution(agents=agents,
                                                                                      coverage=coverage_map,
                                                                                      M=max_cover),
                    "agent_overhead_actions": overhead_actions,
                    "agent_net_contribution": net_contributions,
                    "sys_convergence_iteration": data[-1]["iteration"]+1,
                    "output_file_UUID": save_files_per_trial[trial]
                })

            filename = os.path.join(JSON_SAVE_PATH, "sim_summaries", algo,
                                    f"{uuid.uuid4()}.json")
            DataCollector.export_to_json(sim_summary, filename)

    @staticmethod
    def get_best_system_iter(data):
        """
        Find the simulation iteration with the maximum score over all iterations.

        Attributes:
            data: all iteration data for a single trial
        Returns:
            best_system_ter: iteration where the best system score was achieved
        """
        max_score = 0
        best_system_iter = None
        for iteration, iteration_data in enumerate(data):
            system = iteration_data["system"]
            sim_score = system.system_score()
            if max_score < sim_score:
                max_score = sim_score
                best_system_iter = system, iteration + 1
        return best_system_iter

    @staticmethod
    def calculate_overhead_net_contribution(data):
        """
        Calculate how many agent actions did not increase overall score (overhead).
        Calculate net contribution of agents. (sum of how all actions have changed overall system score)

        Attributes:
            data: all iteration data for a single trial
        Returns:
            agent_overhead_counts: how many actions did not increase overall system score
            agent_net_contributions: agents net contribution to system score
        """
        initial_system = data[0].get("system")
        agent_actions = {agent.id: agent.current_action for agent in initial_system.agents}
        agent_overhead_action_counts = {agent.id: 0 for agent in initial_system.agents}
        agent_net_contributions = {agent.id: 0 for agent in initial_system.agents}
        prev_sys_score = initial_system.system_score()

        for iteration_data in data[1:]:
            system = iteration_data.get("system")
            sys_score = system.system_score()
            for agent in system.agents:
                curr_action = {r.id for r in agent.current_action}
                prev_action = {r.id for r in agent_actions[agent.id]}

                if curr_action != prev_action and prev_sys_score >= sys_score:
                    agent_overhead_action_counts[agent.id] += 1

                if curr_action != prev_action:
                    agent_net_contributions[agent.id] += (sys_score - prev_sys_score)

                agent_actions[agent.id] = agent.current_action

            prev_sys_score = sys_score

        return agent_overhead_action_counts, agent_net_contributions

    @staticmethod
    def calculate_resource_popularity(data):
        """
        Find the count for how many times an agent has selected a resource.

        Attributes:
            data: all iteration data for a single trial
        Returns:
            resource_popularity: count per resource of how many times an agent chose to cover it.
        """
        # get initial agent actions
        agent_actions = {agent.id: agent.current_action for agent in data[0].get("system").agents}
        resource_popularity = {r.id: 0 for r in data[0].get("system").resources}

        for iteration_data in data:
            system = iteration_data.get("system")
            for agent in system.agents:
                # find resources that have just been selected by an agent, ignore if they choose the same action, or
                # action was not updated this iteration
                curr_action = {r.id for r in agent.current_action}
                prev_action = {r.id for r in agent_actions[agent.id]}

                for resource in agent.current_action:
                    if curr_action != prev_action:
                        resource_popularity[resource.id] += 1

                agent_actions[agent.id] = agent.current_action

        return resource_popularity

    @staticmethod
    def calculate_agent_contribution(agents, coverage, M):
        """
        Find the final contribution per agent given the resources they chose to cover.

        Attributes:
            agents: final state of the agents
            coverage: final choices of the agents
            M: max cover value for this simulation
        Returns:
            agent_action_contributions: value given to system score given agent allocations
        """
        agent_action_contributions = {agent.id: 0 for agent in agents}

        for agent in agents:
            system_contribution = sum(
                r.value / coverage[r.id] for r in agent.current_action if r.id in coverage and coverage[r.id] >= M)
            agent_action_contributions[agent.id] = system_contribution
        return agent_action_contributions

    @staticmethod
    def count_agent_actions(data):
        """
        Calculate per agent how many times they have "changed their minds"

        Attributes:
            data: all iteration data for a single trial
        Returns:
            agent_action_counts: count of how many times the agent changed their current action
        """
        # get initial agent actions
        agent_actions = {agent.id: agent.current_action for agent in data[0].get("system").agents}
        agent_action_counts = {agent.id: 0 for agent in data[0].get("system").agents}

        for iteration_data in data[1:]:
            system = iteration_data.get("system")

            for agent in system.agents:
                # if the agent has changed their mind from the last iteration, count it
                curr_action = {r.id for r in agent.current_action}
                prev_action = {r.id for r in agent_actions[agent.id]}

                if curr_action != prev_action:
                    agent_action_counts[agent.id] += 1

                agent_actions[agent.id] = agent.current_action

        return agent_action_counts

    @staticmethod
    def export_to_json(data, file_name):
        """
        Find the final contribution per agent given the resources they chose to cover.

        Attributes:
            data: full simulation summary data
            file_name: file name
        Returns:
            None
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as file:
            json.dump(data, file, indent=4)
