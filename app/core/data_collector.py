import json
import pprint
from copy import deepcopy
from typing import final

from app.core.algorithms import brute_force
from app.utils.common_utils import format_agent_data

class DataCollector:
    def __init__(self):
        self.results = []

    # metrics
    # 1. Brute Force Score vs Attained score
    # 2. Agent resource changes, action count
    # 3. Percentage of resources covered
    # 4. Average agents per resource
    # 5. over coverage of agents (agents on r > M)
    # do under coverage as well
    # 6. resource volatility, how often was this resource covered or not
    # 7. agent contribution (algorithmic fairness)
    # 9. agent action utilization (how many actions are "unused", not contributing to overall score)
    # 11. overhead actions, how many actions did not improve system score?
    # 12. system convergence time, use metric 8
    # 13. agent load balance, is one agent acting more often than others?
    # 14. agent net contribution? +,- to score as sim is running

    def log(self, trial, iteration, system):
        self.results.append({
            "trial": trial,
            "iteration": iteration,
            "system": system
        })

    # TODO:: Unit test!!
    def summarize_results(self, file_names):
        results_by_trial = {}
        for entry in self.results:
            trial_num = entry["trial"]
            if trial_num not in results_by_trial:
                results_by_trial[trial_num] = []
            results_by_trial[trial_num].append(entry)

        # get trial data to construct sim_summary dict
        unique_trials = sorted(set(entry["trial"] for entry in self.results))
        sim_summary = {trial: {} for trial in unique_trials}

        for trial, data in results_by_trial.items():
            original_system = [entry for entry in data if entry["iteration"] == 1][0].get("system")
            # clear any agent allocation, should already be cleared
            for agent in original_system.agents:
                agent.current_action = set()

            final_system = results_by_trial[trial][-1]["system"]

            brute_force_score = brute_force(deepcopy(original_system))[0]
            if brute_force_score ==0:
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

            agent_actions = {}
            for agent in agents:
                res_list = []
                for resource in agent.current_action:
                    res_list.append((resource.id, resource.value))
                agent_actions[agent.id] = res_list

            resources_covered = sum(1 for resource in resources if coverage_map[resource.id] >= max_cover)
            over_coverage_map = {resource.id:coverage_map[resource.id] for resource in resources if coverage_map[resource.id] > max_cover}
            overhead_actions, net_contributions = DataCollector.calculate_overhead_net_contribution(data=data)

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
                "grade": str(sim_score/brute_force_score)+"%",
                "agent_total_actions": DataCollector.count_agent_actions(data=data),
                "resource_popularity":DataCollector.calculate_resource_popularity(data=data),
                "agent_contributions": DataCollector.calculate_agent_contribution(agents=agents,
                                                                                  coverage=coverage_map, M=max_cover),
                "agent_overhead_actions":overhead_actions,
                "agent_net_contribution":net_contributions,
                "sys_convergence_time":0,
                "output_file_UUID": file_names[trial-1]
            })

            output_str = "sim_results_file_" + file_names[trial-1][-36:]
            # DataCollector.save_to_json(sim_summary, output_str)
            with open(output_str, "w") as file:
                json.dump(sim_summary, file, indent=4)

    @staticmethod
    def calculate_overhead_net_contribution(data):
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

    #TODO:Unit test!!
    @staticmethod
    def calculate_resource_popularity(data):
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

    #TODO:Unit test!!
    @staticmethod
    def calculate_agent_contribution(agents, coverage, M):
        agent_action_contributions = {agent.id: 0 for agent in agents}

        for agent in agents:
            # TODO:: "/coverage[r.id]", add this after r.value potentially
            system_contribution = sum(r.value for r in agent.current_action if r.id in coverage and coverage[r.id] >= M)
            agent_action_contributions[agent.id] = system_contribution
        return agent_action_contributions

    #TODO:: Unit test!!
    @staticmethod
    def count_agent_actions(data):
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
    def save_to_json(data, filename="simulation_results.json"):
        json_data = json.dumps(data)
        formatted_json = pprint.pformat(json_data, sort_dicts=False)

        with open(filename, "w") as f:
            f.write(formatted_json)

    @staticmethod
    def format_json(json_string, indent=2):
        data = json.loads(json_string)
        return json.dumps(data, indent=indent)