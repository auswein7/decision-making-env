from app.utils.logger import Logger

logger = Logger.get_logger()

def log_system_properties(system, trial_num):
    """
    Export information to the 'experiment.log' file

    :param system: system for the given trial
    :param trial_num: current running trial
    :return: none
    """
    logger.info(f"Trial {trial_num+1}")
    logger.info(f"Max Cover {system.M}")
    logger.info(f"Num Resources: {len(system.resources)}")

    out_list = []
    for resource in system.resources:
        out_list.append((resource.id, resource.value))
    logger.info(f"Resource List: {out_list}")

    logger.info(f"Num Agents: {len(system.agents)}")
    for agent in system.agents:
        out_list = []
        for subset in agent.action_set:
            sub_list = []
            for resource in subset:
                output_tuple = (resource.id, resource.value)
                sub_list.append(output_tuple)
            out_list.append(sub_list)
        logger.info(f"Agent {agent.id} Action Set: {out_list}")

def log_agent_allocation(system):
    """
    Export information to the 'experiment.log' file. This data is collected at the end of the run
    and shows the actions the agents chose.

    :param system: system state after trial has completed
    :return: none
    """
    logger.info(f"Simulation score {system.system_score()}")
    for agent in system.agents:
        out_list = []
        for resource in agent.current_action:
            out_list.append((resource.id, resource.value))

        logger.info(f"Agent {agent.id} Covers: {out_list}")

    logger.info(f"System resource coverage: {system.resource_coverage}")