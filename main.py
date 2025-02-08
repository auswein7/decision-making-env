from app.utils.arg_parser import read_app_properties
from app.experiments.experiments import run_experiments
from app.experiments.experiments import run_from_json


# TODO:: Set up pipeline to auto run unit-tests on push

def main(args):
    load_from_config = bool(args.load_from_config)
    if load_from_config:
        run_from_json(args)
    else:
        run_experiments(args)


if __name__ == "__main__":
    cmd_args = read_app_properties()
    main(cmd_args)
