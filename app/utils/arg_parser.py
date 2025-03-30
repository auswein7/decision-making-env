import argparse
import configparser

# TODO:: fill out all help documentation
def read_app_properties():
    """
    Function allows all arguments for system creation to be passed through CMD. Default values are pulled from
    'application.properties' file.

    :return: args
    """
    config = configparser.ConfigParser()
    config.read('application.properties')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trial_repetitions",
        type=int,
        default=int(config['DEFAULT'].get('trial_repetitions', "0")),
        help="Amount of trials to repeat with a set configuration."
    )

    parser.add_argument(
        "--iterations_per_trial",
        type=int,
        default=int(config['DEFAULT'].get('iterations_per_trial', "0")),
        help="The number of iterations to run in each trial"
    )

    parser.add_argument(
        "--num_resources",
        type=int,
        default=int(config['DEFAULT'].get('num_resources', "0")),
        help="The number of resources to load into each System"
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=int(config['DEFAULT'].get('num_agents', "0")),
        help="The number of agents to load into each System"
    )

    parser.add_argument(
        "--max_cover",
        type=int,
        default=int(config['DEFAULT'].get('max_cover', "0")),
        help="The number of agents needed to cover a resource and gain its value"
    )

    parser.add_argument(
        "--resource_val_lb",
        type=int,
        default=int(config['DEFAULT'].get('resource_val_lb', "0")),
        help="The lower bound on the value of a resource"
    )

    parser.add_argument(
        "--resource_val_ub",
        type=int,
        default=int(config['DEFAULT'].get('resource_val_ub', "0")),
        help="The upper bound on the value of a resource"
    )

    parser.add_argument(
        "--agent_action_len_lb",
        type=int,
        default=int(config['DEFAULT'].get('agent_action_len_lb', "0")),
        help="The lower bound on how total subsets are contained in the agents action set"
    )

    parser.add_argument(
        "--agent_action_len_ub",
        type=int,
        default=int(config['DEFAULT'].get('agent_action_len_ub', "0")),
        help="The upper bound on how total subsets are contained in the agents action set "
    )

    parser.add_argument(
        "--agent_subset_len_lb",
        type=int,
        default=int(config['DEFAULT'].get('agent_subset_len_lb', "0")),
        help="The lower bound on how many resources can be covered in one subset of agent action"
    )

    parser.add_argument(
        "--agent_subset_len_ub",
        type=int,
        default=int(config['DEFAULT'].get('agent_subset_len_ub', "0")),
        help="The upper bound on how many resources can be covered in one subset of agent action"
    )

    parser.add_argument(
        "--load_from_config",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('load_from_config', "False").lower() in ['true', '1', 'yes'],
        help="Boolean to load from predefined json, or generate random system configurations"
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default=str(config['DEFAULT'].get('distribution', "")),
        help="The probability distribution to use for agent decisions"
    )

    parser.add_argument(
        "--utility",
        type=str,
        default=str(config['DEFAULT'].get('utility', "")),
        help="How will the agents weight the value of each resource relative to other agents presence, comma separated"
             "list of utility functions to run in sequentially."
    )

    parser.add_argument(
        "--system_convergence_iter",
        type=int,
        default=int(config['DEFAULT'].get('system_convergence_iter', "0")),
        help=""
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=float(config['DEFAULT'].get('beta', "0.0")),
        help="Beta value for the approximate best response algorithm"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=float(config['DEFAULT'].get('temperature', "0.0")),
        help="Temperature value for the logit response algorithm"
    )

    parser.add_argument(
        "--best_response_gt",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('best_response_gt', "False").lower() in ['true', '1', 'yes'],
        help=""
    )

    parser.add_argument(
        "--analyze_system",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_system', "False").lower() in ['true', '1', 'yes'],
        help=""
    )

    parser.add_argument(
        "--num_systems",
        type=int,
        default=int(config['DEFAULT'].get('num_systems', "0")),
        help=""
    )

    parser.add_argument(
        "--analyze_beta",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_beta', "False").lower() in ['true', '1', 'yes'],
        help=""
    )

    parser.add_argument(
        "--analyze_temperature",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_temperature', "False").lower() in ['true', '1', 'yes'],
        help=""
    )

    parser.add_argument(
        "--find_optimal_iterations",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('find_optimal_iterations', "False").lower() in ['true', '1', 'yes'],
        help=""
    )

    parser.add_argument(
        "--system_file_uuids",
        type=str,
        default=str(config['DEFAULT'].get('system_file_uuids', "")),
        help=""
    )

    args = parser.parse_args()
    return args
