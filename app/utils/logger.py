import logging


# TODO:: refactor this class to not be a singleton if we want logging in the future, potentially remove logger
class Logger:
    """
    Logger singleton class. Allows for instantiation in any file. Any logging exports to
    'experiment.log' file.

    Attributes:
        _instance: reference to logger instance
    """
    _instance = None

    @classmethod
    def get_logger(cls, name="sim_logger", log_file="experiment.log"):
        if cls._instance is None:
            cls._instance = logging.getLogger(name)
            cls._instance.setLevel(logging.INFO)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('{asctime} {name} {levelname:6s} {message}', style='{')
            fh.setFormatter(formatter)
            cls._instance.addHandler(fh)
        return cls._instance

    # def log_system_properties(self, system, trial_num):
    #     """
    #     Export information to the 'experiment.log' file
    #
    #     :param system: system for the given trial
    #     :param trial_num: current running trial
    #     :return: none
    #     """
    #     logger.info(f"Trial {trial_num}")
    #     logger.info(f"Max Cover {system.M}")
    #     logger.info(f"Num Resources: {len(system.resources)}")
    #
    #     out_list = []
    #     for resource in system.resources:
    #         out_list.append((resource.id, resource.value))
    #     logger.info(f"Resource List: {out_list}")
    #
    #     logger.info(f"Num Agents: {len(system.agents)}")
    #     agent_data = format_agent_data(system.agents)
    #     for a_id, action_set in agent_data.items():
    #         logger.info(f"Agent {a_id} Action Set: {action_set}")
    #
    # def log_agent_allocation(self, system):
    #     """
    #     Export information to the 'experiment.log' file. This data is collected at the end of the run
    #     and shows the actions the agents chose.
    #
    #     :param system: system state after trial has completed
    #     :return: none
    #     """
    #     logger.info(f"Simulation score {system.system_score()}")
    #     for agent in system.agents:
    #         out_list = []
    #         for resource in agent.current_action:
    #             out_list.append((resource.id, resource.value))
    #
    #         logger.info(f"Agent {agent.id} Covers: {out_list}")
    #
    #     logger.info(f"System resource coverage: {system.resource_coverage}")
