from app.experiments.experiments import run_experiments
from app.experiments.experiments import run_from_json
from app.utils.arg_parser import read_app_properties


# TODO:: Set up pipeline to auto run unit-tests on push
# TODO:: make sure using as many built in looping funcs, as well as things like numpy to speed up project

# TODO:: run experiments to find the optimal number of iterations to run for
# TODO:: Define what it means to run an experiment
# TODO:: quantify the difficulty of a system, plot the score as, x axis is easy games -> hard games

def main(args):
    load_from_config = bool(args.load_from_config)
    if load_from_config:
        run_from_json(args)
    else:
        run_experiments(args)


if __name__ == "__main__":
    cmd_args = read_app_properties()
    main(cmd_args)
