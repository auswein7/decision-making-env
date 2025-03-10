import json
import os

from app.utils.common_utils import format_agent_data
from app.utils.constants import JSON_SAVE_PATH


class DataCollector:
    """
    DataCollector class to track all simulation data and compile into a simulation summary json.

    Attributes:
        results: data aggregated through scenario execution.
    """

    def __init__(self, algorithms, uuid_file_map):
        self.results = {algo_name: [] for algo_name in algorithms}
        self.uuid_file_map = uuid_file_map
        self.algorithms = algorithms
        self.logged_sys = False

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
                "resource_values": formatted_resources
            })}

            filename = os.path.join(JSON_SAVE_PATH, "sim_summaries", algorithm,
                                    f"{self.uuid_file_map[algorithm]}.json")

            DataCollector.export_to_json(sys_config_json, filename)

    # TODO:: push to a database, probably mongoDB
    def summarize_results(self, save_file_per_trial, run_args, optimal_score):
        """
        Compile all stored data in self.results.

        Attributes:
            save_file_per_trial: trial num -> system json uuid saved for this trial
        """
        for algo in self.results.keys():
            trial_num = self.results[algo][0]['trial']
            sim_summary = {trial_num: {}}
            final_sys = self.results[algo][-1]["system"]

            agents = final_sys.agents
            sim_score = final_sys.score
            coverage_map = final_sys.resource_coverage
            resources = final_sys.resources
            max_cover = final_sys.M

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

            overhead_actions, net_contributions, agent_action_count = DataCollector.calculate_overhead_net_contribution_actions(data=self.results[algo])
            best_system, best_iteration = DataCollector.get_best_system_iter(data=self.results[algo])

            sim_summary[trial_num] = ({
                "agent_allocations": agent_actions,
                "resource_coverage": coverage_map,
                "resource_coverage_percentage": resources_covered / len(resources),
                "over_covered_resources": over_coverage_map,
                "max_possible_score": optimal_score,
                "simulation_score": sim_score,
                "grade": str((sim_score / optimal_score) * 100) + "%",
                "best_system_score": best_system.score,
                "iteration_of_best_system": best_iteration,
                "agent_total_actions": agent_action_count,
                "resource_popularity": DataCollector.calculate_resource_popularity(data=self.results[algo]),
                "agent_contributions": DataCollector.calculate_agent_contribution(agents=agents,
                                                                                  coverage=coverage_map,
                                                                                  M=max_cover),
                "agent_overhead_actions": overhead_actions,
                "agent_net_contribution": net_contributions,
                "sys_convergence_iteration": self.results[algo][-1]["iteration"]+1,
                "output_file_UUID": save_file_per_trial[trial_num],
                "run_args": run_args[algo],
            })

            filename = os.path.join(JSON_SAVE_PATH, "sim_summaries", algo,
                                    f"{self.uuid_file_map[algo]}.json")

            DataCollector.export_to_json(sim_summary, filename)

        # clear results to save memory
        self.results = {algo_name: [] for algo_name in self.algorithms}

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
            sim_score = system.score
            if max_score < sim_score:
                max_score = sim_score
                best_system_iter = system, iteration + 1
        return best_system_iter

    @staticmethod
    def calculate_overhead_net_contribution_actions(data):
        """
        Calculate how many agent actions did not increase overall score (overhead).
        Calculate net contribution of agents. (sum of how all actions have changed overall system score)

        Attributes:
            data: all iteration data for a single trial
        Returns:
            agent_overhead_counts: how many actions did not increase overall system score
            agent_net_contributions: agents net contribution to system score
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
    def export_to_json(data, file_name):
        """
        Store the simulation summary data over repeated trials.

        Attributes:
            data: Dictionary containing simulation results {trial_number: data}
            file_name: JSON file where the data should be stored
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

        # Iterate over new data entries
        for trial_key, trial_data in data.items():
            trial_key = str(trial_key)
            if trial_key in prev_trials:
                # Find next available key with decimal increments
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


