from app.experiments.experiments import run_experiments

def main():
    num_trials = 1
    num_resources = 30
    num_agents = 20
    m = 2

    run_experiments(num_trials, num_resources, num_agents, m)

if __name__ == "__main__":
    main()