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
        help="The number of iterations to run. 1 iteration = 1 agent wake up."
    )

    parser.add_argument(
        "--num_resources",
        type=int,
        default=int(config['DEFAULT'].get('num_resources', "0")),
        help="The number of resources."
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=int(config['DEFAULT'].get('num_agents', "0")),
        help="The number of agents."
    )

    parser.add_argument(
        "--max_cover",
        type=int,
        default=int(config['DEFAULT'].get('max_cover', "0")),
        help="The number of agents needed to cover a resource."
    )

    parser.add_argument(
        "--resource_val_lb",
        type=int,
        default=int(config['DEFAULT'].get('resource_val_lb', "0")),
        help="The lower bound on the value of a resource."
    )

    parser.add_argument(
        "--resource_val_ub",
        type=int,
        default=int(config['DEFAULT'].get('resource_val_ub', "0")),
        help="The upper bound on the value of a resource."
    )

    parser.add_argument(
        "--agent_action_len_lb",
        type=int,
        default=int(config['DEFAULT'].get('agent_action_len_lb', "0")),
        help="The lower bound on how total subsets are contained in the agents action set."
    )

    parser.add_argument(
        "--agent_action_len_ub",
        type=int,
        default=int(config['DEFAULT'].get('agent_action_len_ub', "0")),
        help="The upper bound on how total subsets are contained in the agents action set."
    )

    parser.add_argument(
        "--agent_subset_len_lb",
        type=int,
        default=int(config['DEFAULT'].get('agent_subset_len_lb', "0")),
        help="The lower bound on how many resources can be covered in one subset of agent action."
    )

    parser.add_argument(
        "--agent_subset_len_ub",
        type=int,
        default=int(config['DEFAULT'].get('agent_subset_len_ub', "0")),
        help="The upper bound on how many resources can be covered in one subset of agent action."
    )

    parser.add_argument(
        "--load_from_config",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('load_from_config', "False").lower() in ['true', '1', 'yes'],
        help="Load from predefined json, or generate random system."
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default=str(config['DEFAULT'].get('distribution', "")),
        help="The probability distribution to use for agent decisions."
    )

    parser.add_argument(
        "--utility",
        type=str,
        default=str(config['DEFAULT'].get('utility', "")),
        help="How will the agents weight the value of each resource."
    )

    parser.add_argument(
        "--system_convergence_iter",
        type=int,
        default=int(config['DEFAULT'].get('system_convergence_iter', "0")),
        help="How many iterations the system score will remain unchanged to exit a run early."
    )

    parser.add_argument(
        "--init_from_optimal",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('init_from_optimal', "False").lower() in ['true', '1', 'yes'],
        help="Agents start at the optimal action choice. Test if they leave the optimal."
    )

    parser.add_argument(
        "--init_from_random",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('init_from_random', "False").lower() in ['true', '1', 'yes'],
        help="All agents start with a random action selected."
    )

    parser.add_argument(
        "--generate_graphs",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('generate_graphs', "False").lower() in ['true', '1', 'yes'],
        help="Run in graph creation mode."
    )

    parser.add_argument(
        "--num_graphs",
        type=int,
        default=int(config['DEFAULT'].get('num_systems', "0")),
        help="Numer of graph to create in graph generation mode."
    )

    parser.add_argument(
        "--graph_type",
        type=str,
        default=str(config['DEFAULT'].get('graph_type', "")),
        help="Type of graph to create in graph generation mode."
    )

    parser.add_argument(
        "--coverage_probability",
        type=float,
        default=float(config['DEFAULT'].get('coverage_probability', "0.5")),
        help="Probability an agent will add a graph edge to its action set."
    )

    parser.add_argument(
        "--erdos_prob",
        type=float,
        default=float(config['DEFAULT'].get('erdos_prob', "0.1")),
        help="Probability to add an edge in erdos renyi graph creation."
    )

    parser.add_argument(
        "--geo_radius",
        type=float,
        default=float(config['DEFAULT'].get('geo_radius', "0.1")),
        help="r variable for random geometric bipartite graph generation."
    )

    parser.add_argument(
        "--ws_beta",
        type=float,
        default=float(config['DEFAULT'].get('ws_beta', "0.1")),
        help="Beta value for watts-strogatz graph generation."
    )

    parser.add_argument(
        "--ws_k",
        type=int,
        default=int(config['DEFAULT'].get('ws_k', "2")),
        help="K neighbors for watts-strogatz graph generation."
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=float(config['DEFAULT'].get('beta', "0.0")),
        help="Beta value for the approximate best response algorithm."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=float(config['DEFAULT'].get('temperature', "0.0")),
        help="Temperature value for the logit response algorithm."
    )

    parser.add_argument(
        "--best_response_gt",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('best_response_gt', "False").lower() in ['true', '1', 'yes'],
        help="Output best response distribution score on all plots."
    )

    parser.add_argument(
        "--analyze_system",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_system', "False").lower() in ['true', '1', 'yes'],
        help="Run in system analysis mode."
    )

    parser.add_argument(
        "--num_systems",
        type=int,
        default=int(config['DEFAULT'].get('num_systems', "0")),
        help="Numer of systems to create when running in system analysis mode."
    )

    parser.add_argument(
        "--analyze_beta",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_beta', "False").lower() in ['true', '1', 'yes'],
        help="Run in parameter analysis mode, analyzing beta."
    )

    parser.add_argument(
        "--analyze_temperature",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('analyze_temperature', "False").lower() in ['true', '1', 'yes'],
        help="Run in parameter analysis mode, analyzing temperature."
    )

    parser.add_argument(
        "--find_optimal_iterations",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=config['DEFAULT'].get('find_optimal_iterations', "False").lower() in ['true', '1', 'yes'],
        help="Run in optimal iterations mode, testing system convergence."
    )

    parser.add_argument(
        "--system_file_uuids",
        type=str,
        default=str(config['DEFAULT'].get('system_file_uuids', "")),
        help="List of system_ids to be loaded."
    )

    args = parser.parse_args()
    return args
