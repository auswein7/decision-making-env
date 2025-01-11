from app.utils.arg_parser import read_app_properties
from app.experiments.experiments import run_experiments

def main(num_trials, num_resources, num_agents, max_cover):
    run_experiments(num_trials, num_resources, num_agents, max_cover)

if __name__ == "__main__":
    args = read_app_properties()
    main(args.num_trials, args.num_resources, args.num_agents, args.max_cover)