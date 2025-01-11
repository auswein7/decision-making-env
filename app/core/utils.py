def system_utility(agent, action, system):
    agent.current_action = action
    return system.system_score()

def log_system_properties(system, trial_num):
    print(f"\nTrial {trial_num+1} beginning with following properties:\n")

    print(f"Num Resources: {len(system.resources)}")
    out_list = []
    for resource in system.resources:
        out_list.append((resource.id, resource.value))
    print(f"Resource List: {out_list}")

    print(f"Num Agents: {len(system.agents)}")
    for agent in system.agents:
        out_list = []
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                output_tuple = (resource.id, resource.value)
                sub_list.append(output_tuple)
            out_list.append(sub_list)
        print(f"Agent {agent.id} Action Set: {out_list}")

def log_agent_allocation(system):
    print(f"\nSimulation terminated with system score {system.system_score()}\n")
    print(f"Agent actions:\n")
    for agent in system.agents:
        out_list = []
        for resource in agent.current_action:
            out_list.append((resource.id, resource.value))

        print(f"Agent {agent.id} Covers: {out_list}")

    print(f"System resouce coverage: {system.resource_coverage}")
