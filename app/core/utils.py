def system_utility(agent, action, system):
    """System-level scoring."""
    original_action = agent.current_action
    agent.current_action = action

    score = system.system_score()

    # Revert the agent's action after calculation
    agent.current_action = original_action

    return score

def log_system_properties(system, trial_num):
    print(f"Trial: {trial_num+1} beginning with following properties:")

    print(f"Num Resources: {len(system.resources)}")
    out_list = []
    for resource in system.resources:
        out_list.append((resource.id, resource.value))
    print(f"Resource List: {out_list}")

    print(f"Num Agents: {len(system.agents)}")
    for agent in system.agents:
        print(f"Agent ID: {agent.id}")
        out_list = []
        for count, subset in enumerate(agent.action_set):
            sub_list = []
            for resource in subset:
                output_tuple = (resource.id, resource.value)
                sub_list.append(output_tuple)
            out_list.append(sub_list)
        print(f"Agent Action Set: {out_list}")


