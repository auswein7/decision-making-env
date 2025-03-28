from app.runner.experiments import run_experiments
from app.runner.experiments import run_from_json
from app.utils.arg_parser import read_app_properties


# TODO:: Set up pipeline to auto run unit-tests on push
# TODO:: make sure using as many built in looping funcs, as well as things like numpy to speed up project

# TODO:: run runner to find the optimal number of iterations to run for
# TODO:: Define what it means to run an experiment
# TODO:: quantify the difficulty of a system, plot the score as, x axis is easy games -> hard games

def main(args):
    load_from_config = bool(args.load_from_config)
    if load_from_config:
        run_from_json(args)
    else:
        run_experiments(args)


# TODO:: IMPLEMENT
def validate_run(args):
    if args['analyze_system'] and len(args['utility_functions']) != 1:
        raise ValueError("System analysis must be run with exactly one utility function.")
    if args['analyze_beta'] and 'approximate_best_response' not in args['distribution']:
        raise ValueError("Beta analysis requires the 'approximate_best_response' distribution.")


if __name__ == "__main__":
    cmd_args = read_app_properties()
    main(cmd_args)
