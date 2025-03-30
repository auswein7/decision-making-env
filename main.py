from app.runner.run_experiments import filter_run
from app.utils.arg_parser import read_app_properties
from app.utils.constants import VALID_DIST_NAMES, VALID_UTIL_NAMES

# TODO:: Define what it means to run an experiment
def main(args):
    filter_run(args)


def validate_run(args):
    # System config checks
    if any([
        args.num_resources < 0,
        args.num_agents < 0,
        args.max_cover > args.num_agents,
        args.resource_val_lb > args.resource_val_ub,
        args.agent_action_len_lb > args.agent_action_len_ub,
        args.agent_subset_len_lb > args.agent_subset_len_ub,
        args.num_systems < 0,
        args.trial_repetitions < 0
    ]):
        print("Infeasible system configuration, check input arguments.")
        return False

    # Define running modes
    mode_1 = args.analyze_system
    mode_2 = args.find_optimal_iterations
    mode_3_flags = [
        args.analyze_temperature,
        args.analyze_beta,
        # extend list here if adding more param analysis in future
    ]

    active_modes = sum([
        bool(mode_1),
        bool(mode_2),
        any(mode_3_flags)
    ])

    # Utility validation
    utility_list = args.utility.split(',')
    if mode_1 and len(utility_list) > 1:
        print("Can select only one utility function for system analysis runs.")
        return False

    if mode_2 and len(utility_list) > 1:
        print("Can select only one utility function for opt iteration runs.")
        return False

    # Distribution validation
    dist_list = args.distribution.split(',')
    if mode_1 and len(dist_list) > 1:
        print("Can select only one distribution function for system analysis runs.")
        return False

    if mode_2 and len(dist_list) > 1:
        print("Can select only one distribution function for opt iteration runs.")
        return False

    if any(mode_3_flags) and len(args.system_file_uuids.split(',')) > 1:
        print("Can only load one previous model for parameter analysis runs.")

    if active_modes > 1:
        print("Cannot run in multiple active modes.")
        return False

    # Distribution and utility function validity
    invalid_dists = [dist for dist in dist_list if dist not in VALID_DIST_NAMES]
    if invalid_dists:
        print("Unknown distribution name.")
        return False

    invalid_utils = [util for util in utility_list if util not in VALID_UTIL_NAMES]
    if invalid_utils:
        print(f"Invalid utility functions: {', '.join(invalid_utils)}")
        return False

    # Parameter bounds
    if args.beta < 0:
        print("Beta value cannot be negative.")
        return False

    if args.temperature < 0:
        print("Temperature value cannot be negative.")
        return False

    return True

if __name__ == "__main__":
    cmd_args = read_app_properties()
    if validate_run(cmd_args):
        main(cmd_args)
    else:
        print("Invalid run configuration given, exiting...")
