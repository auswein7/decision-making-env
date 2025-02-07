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

    def __init__(self):
        self.results = []

    def log(self, trial, iteration, system, algorithm):
        """
        Log the current state during every iteration. Called trial * iteration times throughout a run.

        Attributes:
            trial: trial number for this simulation.
            iteration: iteration number for this trial.
            system: configuration at trial/iteration
            algorithm: algorithm used for entire simulation
        """
        self.results.append({
            "trial": trial,
            "iteration": iteration,
            "system": system,
            "algorithm": algorithm
        })

    # TODO:: Unit test!!
    def summarize_results(self, file_names):
        """
        Compile all stored data in self.results.

        Attributes:
            file_names: list of file names to link simulation results to exported system json file.
        """
        # format results into dictionary, keyed by trial number
        results_by_trial = {}
        for entry in self.results:
            trial_num = entry["trial"]
            if trial_num not in results_by_trial:
                results_by_trial[trial_num] = []
            results_by_trial[trial_num].append(entry)

        # set up output data, keyed by trial number
        unique_trials = sorted(set(entry["trial"] for entry in self.results))
        sim_summary = {trial: {} for trial in unique_trials}

        for trial, data in results_by_trial.items():
            # get the system from trial 1 iteration 1
            original_system = [entry for entry in data if entry["iteration"] == 1][0].get("system")

            for agent in original_system.agents:
                agent.current_action = set()

            # get the system from final trial final iteration
            final_system = results_by_trial[trial][-1]["system"]

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
            best_system, best_iteration = DataCollector.get_best_system_config(data=data)

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
                                                                                  coverage=coverage_map, M=max_cover),
                "agent_overhead_actions": overhead_actions,
                "agent_net_contribution": net_contributions,
                "sys_convergence_time": 0,
                "sys_score_vs_beta": DataCollector.calculate_system_score_vs_beta(data=data),
                "output_file_UUID": file_names[trial - 1]
            })

            filename = os.path.join(JSON_SAVE_PATH, "sim_summaries", self.results[0]["algorithm"],
                                    f"{uuid.uuid4()}.json")
            DataCollector.export_to_json(sim_summary, filename)

    @staticmethod
    def get_best_system_config(data):
        """
        Find the system configuration that achieved the maximum score over all iterations.

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

    # TODO:: FINSIH IMPLEMENTATION
    @staticmethod
    def calculate_system_score_vs_beta(data):
        """
        Construct x/y dimensional data comparing beta's effect on system score.

        Attributes:
            data: all iteration data for a single trial
        Returns:
            beta_sys_score: x/y dimensional data
        """
        # Calculated at the trial level
        # contruct a dictionary of trial: average_system_score
        avg_sys_score = 0
        count = 0
        for iteration_data in data:
            system = iteration_data["system"]
            avg_sys_score += system.system_score()
            count += 1

        avg_sys_score = avg_sys_score / count

        return avg_sys_score

    # TODO:: FINISH IMPLEMENTATION
    @staticmethod
    def calculate_system_convergence(data, conv_iter=0):
        """
        Determine if the agents choices, and system score have converged.

        Attributes:
            data: all iteration data for a single trial
            conv_iter: iterations threshold, if system has not changed behavior in the past conv_iter iterations
            the system has converged
        Returns:
            conv_iter: iteration of convergence, -1 if have not converged
        """
        # TODO:: lookup markov chain, converging to a stationary distribution
        # average over past N iterations
        state_map = {}

        for iteration_data in data:
            continue
        return -1

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

    # TODO:Unit test!!
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

    # TODO:Unit test!!
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
            # TODO:: "/coverage[r.id]", add this after r.value potentially
            system_contribution = sum(r.value for r in agent.current_action if r.id in coverage and coverage[r.id] >= M)
            agent_action_contributions[agent.id] = system_contribution
        return agent_action_contributions

    # TODO:: Unit test!!
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
