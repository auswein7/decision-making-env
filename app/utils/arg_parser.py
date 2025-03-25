import argparse
import configparser

# TODO:: REMOVE GENETIC PARAMS, REORDER TO REFLECT APP.PROPS ORDER

def read_app_properties():
    """
    Function allows all arguments for system creation to be passed through CMD. Default values are pulled from
    'application.properties' file.

    :return: args
    """
    config = configparser.ConfigParser()
    config.read('application.properties')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # Add arguments with default values from the properties file
    parser.add_argument(
        "--num_trials",
        type=int,
        default=int(config['DEFAULT'].get('num_trials', "0")),
        help="The number of random system configurations to test"
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
        help="Boolean to load from predefined json, or generate random system"
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
        "--analyze_beta",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_beta', "False").lower() in ['true', '1', 'yes'],
        help="Boolean to generate simulation graphics after each trial"
    )

    parser.add_argument(
        "--analyze_temperature",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_temperature', "False").lower() in ['true', '1', 'yes'],
        help="Boolean to generate simulation graphics after each trial"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=str(config['DEFAULT'].get('algorithm', "")),
        help="The algorithm to use for agent allocation"
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default=str(config['DEFAULT'].get('distribution', "")),
        help="The probability distribution to use for agent decisions"
    )

    parser.add_argument(
        "--utility_functions",
        type=str,
        default=str(config['DEFAULT'].get('utility_functions', "")),
        help="How will the agents weight the value of each resource relative to other agents presence, comma separated"
    )

    parser.add_argument(
        "--system_convergence_iter",
        type=int,
        default=int(config['DEFAULT'].get('system_convergence_iter', "0")),
        help="The upper bound on how many resources can be covered in one subset of agent action"
    )

    parser.add_argument(
        "--trial_repetitions",
        type=int,
        default=int(config['DEFAULT'].get('trial_repetitions', "1")),
        help="Amount of trials to repeat with a set parameter value. Used in custom run for parameter analysis"
             "averages."
    )

    args = parser.parse_args()
    return args
