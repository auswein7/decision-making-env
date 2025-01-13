import argparse
import configparser

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
        default=int(config['DEFAULT'].get('num_trials', "")),
        help="The number of random system configurations to test"
    )

    parser.add_argument(
        "--num_resources",
        type=int,
        default=int(config['DEFAULT'].get('num_resources', "")),
        help="The number of resources to load into each System"
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=int(config['DEFAULT'].get('num_agents', "")),
        help="The number of agents to load into each System"
    )

    parser.add_argument(
        "--max_cover",
        type=int,
        default=int(config['DEFAULT'].get('max_cover', "")),
        help="The number of agents needed to cover a resource and gain its value"
    )

    parser.add_argument(
        "--resource_val_lb",
        type=int,
        default=int(config['DEFAULT'].get('resource_val_lb', "")),
        help="The lower bound on the value of a resource"
    )

    parser.add_argument(
        "--resource_val_ub",
        type=int,
        default=int(config['DEFAULT'].get('resource_val_ub', "")),
        help="The upper bound on the value of a resource"
    )

    parser.add_argument(
        "--agent_subset_len_lb",
        type=int,
        default=int(config['DEFAULT'].get('agent_subset_len_lb', "")),
        help="The lower bound on how many resources can be covered in one subset of agent action"
    )

    parser.add_argument(
        "--agent_subset_len_ub",
        type=int,
        default=int(config['DEFAULT'].get('agent_subset_len_ub', "")),
        help="The upper bound on how many resources can be covered in one subset of agent action"
    )

    args = parser.parse_args()
    return args