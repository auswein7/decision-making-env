import argparse
import configparser

def read_app_properties():
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

    args = parser.parse_args()
    return args