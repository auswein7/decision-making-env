from app.utils.arg_parser import read_app_properties
from app.experiments.experiments import run_experiments


# TODO:: Set up pipeline to auto run unit-tests on push

def main(args):
    run_experiments(args)


if __name__ == "__main__":
    cmd_args = read_app_properties()
    main(cmd_args)
