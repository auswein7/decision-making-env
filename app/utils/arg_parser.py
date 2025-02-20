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
        "--generate_graphics",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('generate_graphics', "False").lower() in ['true', '1', 'yes'],
        help="Boolean to generate simulation graphics after each trial"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=str(config['DEFAULT'].get('algorithm', "")),
        help="The algorithm to use for agent allocation"
    )

    parser.add_argument(
        "--system_convergence_iter",
        type=int,
        default=int(config['DEFAULT'].get('system_convergence_iter', "0")),
        help="The upper bound on how many resources can be covered in one subset of agent action"
    )

    parser.add_argument(
        "--population_size",
        type=int,
        default=int(config['DEFAULT'].get('population_size', "0")),
        help="Size of populations per generation of GA"
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=int(config['DEFAULT'].get('generations', "0")),
        help="Generations (iterations) to use for GA"
    )

    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=float(config['DEFAULT'].get('mutation_rate', "0.0")),
        help="Rate at which agent actions will by altered within a member of a population"
    )

    parser.add_argument(
        "--tournament_k",
        type=int,
        default=int(config['DEFAULT'].get('tournament_k', "0")),
        help="Percentage of individuals from population to compete in tournament selection"
    )

    parser.add_argument(
        "--selection_pressure",
        type=float,
        default=float(config['DEFAULT'].get('selection_pressure', "0.0")),
        help="Likelihood of individuals to be selected to compete in a tournament, 1 means consider "
             "only the most fit individuals, 0 means select random individuals for tournament selection"
    )

    parser.add_argument(
        "--num_parents",
        type=int,
        default=int(config['DEFAULT'].get('num_parents', "0")),
        help="Number of parent individuals to use for crossover in ga"
    )

    parser.add_argument(
        "--generational_size",
        type=float,
        default=float(config['DEFAULT'].get('generational_size', "0.9")),
        help="Size of new generations in ga, percentage of original population size, 1 maintains original size"
    )


    parser.add_argument(
        "--k_crossover",
        type=int,
        default=int(config['DEFAULT'].get('k_crossover', "0")),
        help="Amount of partitions to make in parent agent actions when forming new generation of individuals"
    )


    parser.add_argument(
        "--batch_wake_up",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('batch_wake_up', "False").lower() in ['true', '1', 'yes'],
        help="Boolean to generate simulation graphics after each trial"
    )

    args = parser.parse_args()
    return args
