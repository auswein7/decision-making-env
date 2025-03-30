import json
import os

from app.utils.common_utils import format_agent_data
from app.utils.constants import JSON_SAVE_PATH


class DataCollector:
    """
    DataCollector class to track all simulation data and compile into a simulation summary json.

    Attributes:
        results: data aggregated through scenario execution.
        sim_sum_uuid: uuid of sim_summary output file.
        data_key: key of key attributes to index data, and filenames
        logged_sys: if initial sys configuration has been logged
    """

    def __init__(self, data_key, sim_sum_uuid):
        self.results = {}
        self.sim_sum_uuid = sim_sum_uuid
        self.data_key = data_key
        self.logged_sys = False

    def log(self, trial, iteration, system, data_key):
        """
        Log initial system state.

        Attributes:
            trial: trial number for this simulation.
            iteration: iteration number for this trial.
            system: configuration at trial/iteration
            data_key: data_key used for index into results dict
        """
        if data_key not in self.results:
            self.results[data_key] = []

        self.results[data_key].append({
            "trial": trial,
            "iteration": iteration,
            "system": system
        })

        if not self.logged_sys:
            self.logged_sys = True

            formatted_resources = []
            for resource in system.resources:
                formatted_resources.append((resource.id, resource.value))

            sys_config_json = {"System_Config": ({
                "max_cover": system.M,
                "num_agents": len(system.agents),
                "agent_ids": [a.id for a in system.agents],
                "action_sets": format_agent_data(system.agents),
                "num_resources": len(system.resources),
                "resource_values": formatted_resources,
                "coverable_resources": DataCollector.get_coverable_resources(system.agents, system.resources, system.M),
                "system_id": system.id
            })}

            filename = os.path.join(JSON_SAVE_PATH, "sim_summaries", data_key,
                                    f"{self.sim_sum_uuid[data_key]}.json")

            DataCollector.export_to_json(sys_config_json, filename)

    def summarize_results(self, save_file, run_args, optimal_score, avg_score):
        """
        Compile all stored data in self.results.

        Attributes:
            save_file: uuid of saved system json
            run_args: arguments passed to run_trial in trial_runner.py
            optimal_score: optimal score of the system
            avg_score: average score of the last 20% of iterations_per_trial
        """
        for data_key in self.results.keys():
            trial_num = self.results[data_key][0]['trial']
            sim_summary = {trial_num: {}}
            final_sys = self.results[data_key][-1]["system"]

            # pull the final state of agent decisions
            agent_actions = {}
            for agent in final_sys.agents:
                res_list = []
                for resource in agent.current_action:
                    res_list.append((resource.id, resource.value))
                agent_actions[agent.id] = res_list

            resources = final_sys.resources
            coverage_map = final_sys.resource_coverage
            resources_covered = sum(1 for resource in resources if coverage_map[resource.id] >= final_sys.M)
            over_coverage_map = {resource.id: coverage_map[resource.id] for resource in resources if
                                 coverage_map[resource.id] > final_sys.M}

            overhead_actions, net_contributions, agent_action_count = DataCollector.calculate_overhead_net_contribution_actions(
                data=self.results[data_key])
            best_system, best_iteration = DataCollector.get_best_system_iter(data=self.results[data_key])
            sim_summary[trial_num] = ({
                "agent_allocations": agent_actions,
                "resource_coverage": coverage_map,
                "resource_coverage_percentage": resources_covered / len(resources),
                "over_covered_resources": over_coverage_map,
                "max_possible_score": optimal_score,
                "final_sys_score": final_sys.score,
                "avg_score": avg_score,
                "grade": str((avg_score / optimal_score) * 100) + "%",
                "best_system_score": best_system.score,
                "iteration_of_best_system": best_iteration,
                "agent_total_actions": agent_action_count,
                "resource_popularity": DataCollector.calculate_resource_popularity(data=self.results[data_key]),
                "agent_contributions": DataCollector.calculate_agent_contribution(agents=final_sys.agents,
                                                                                  coverage=coverage_map,
                                                                                  M=final_sys.M),
                "agent_overhead_actions": overhead_actions,
                "agent_net_contribution": net_contributions,
                "sys_convergence_iteration": self.results[data_key][-1]["iteration"] + 1,
                "output_file_UUID": save_file,
                "run_args": run_args[data_key],
            })

            filename = os.path.join(JSON_SAVE_PATH, "sim_summaries", data_key,
                                    f"{self.sim_sum_uuid[data_key]}.json")

            DataCollector.export_to_json(sim_summary, filename)

        # clear results to save memory
        self.results = {}

    @staticmethod
    def get_best_system_iter(data):
        """
        Find the iteration with the maximum score over iterations_per_trial.

        Attributes:
            data: iterations_per_trial systems
        Returns:
            best_system_ter: iteration of the best system score
        """
        best_entry = max(
            enumerate(data, start=1),
            key=lambda x: x[1]["system"].score
        )
        system = best_entry[1]["system"]
        iteration = best_entry[0]
        return system, iteration

    @staticmethod
    def get_coverable_resources(agents, resources, M):
        """
        Return list of resources that can possibly be covered.

        Attributes:
            agents: sys agents
            resources: sys resources
            M: max_cover
        Returns:
            coverable_ids: ids of coverable resources
        """
        resource_to_agents = {r: set() for r in resources}

        for agent in agents:
            for action in agent.action_set:
                for r in action:
                    resource_to_agents[r].add(agent.id)

        coverable_ids = []
        for r, agent_set in resource_to_agents.items():
            if len(agent_set) >= M:
                coverable_ids.append(str(r.id))

        return coverable_ids

    @staticmethod
    def calculate_overhead_net_contribution_actions(data):
        """
        Calculate how many agent actions did not increase overall score (overhead).
        Calculate net contribution of agents. (net sum of action scores)

        Attributes:
            data: iterations_per_trial systems
        Returns:
            agent_overhead_counts: how many actions did not increase overall system score
            agent_net_contributions: agents net contribution to system score
            agent_action_counts: total actions taken by agents
        """
        initial_system = data[0]['system']
        agent_actions = {agent.id: agent.current_action for agent in initial_system.agents}
        agent_action_counts = {agent.id: 0 for agent in initial_system.agents}
        agent_overhead_action_counts = {agent.id: 0 for agent in initial_system.agents}
        agent_net_contributions = {agent.id: 0 for agent in initial_system.agents}
        prev_sys_score = initial_system.score

        for iteration_data in data[1:]:
            system = iteration_data['system']
            sys_score = system.score
            for agent in system.agents:
                curr_action = {r.id for r in agent.current_action}
                prev_action = {r.id for r in agent_actions[agent.id]}

                if curr_action != prev_action and prev_sys_score >= sys_score:
                    agent_overhead_action_counts[agent.id] += 1

                if curr_action != prev_action:
                    agent_net_contributions[agent.id] += (sys_score - prev_sys_score)
                    agent_action_counts[agent.id] += 1

                agent_actions[agent.id] = agent.current_action

            prev_sys_score = sys_score

        return agent_overhead_action_counts, agent_net_contributions, agent_action_counts

    @staticmethod
    def calculate_resource_popularity(data):
        """
        Return count of how many times an agent has selected a resource.

        Attributes:
            data: iterations_per_trial systems
        Returns:
            resource_popularity: count per resource of how many times an agent chose to cover it.
        """
        agent_actions = {agent.id: agent.current_action for agent in data[0].get("system").agents}
        resource_popularity = {r.id: 0 for r in data[0].get("system").resources}

        for iteration_data in data:
            system = iteration_data.get("system")
            for agent in system.agents:
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
        Get final value added to system by agent. Divided evenly to all agents on a resource.

        Attributes:
            agents: final state of the agents
            coverage: final choices of the agents
            M: max_cover
        Returns:
            agent_action_contributions: agent contribution to system score
        """
        agent_action_contributions = {agent.id: 0 for agent in agents}

        for agent in agents:
            system_contribution = sum(
                r.value / coverage[r.id] for r in agent.current_action if r.id in coverage and coverage[r.id] >= M)
            agent_action_contributions[agent.id] = system_contribution
        return agent_action_contributions

    @staticmethod
    def export_to_json(data, file_name):
        """
        Store the simulation summary data.

        Attributes:
            data: Dictionary containing simulation results {trial_number: data}
            file_name: JSON output filename
        Returns:
            None
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                try:
                    prev_trials = json.load(file)
                except json.JSONDecodeError:
                    prev_trials = {}
        else:
            prev_trials = {}

        for trial_key, trial_data in data.items():
            trial_key = str(trial_key)
            if trial_key in prev_trials:
                # Find next available key with decimal increments (for trial_repetitions)
                suffix = 1
                new_key = f"{trial_key}.{suffix}"
                while new_key in prev_trials:
                    suffix += 1
                    new_key = f"{trial_key}.{suffix}"
                prev_trials[new_key] = trial_data
            else:
                prev_trials[trial_key] = trial_data

        # Write back to file
        with open(file_name, "w") as file:
            json.dump(prev_trials, file, indent=4)
