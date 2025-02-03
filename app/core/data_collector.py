import json

class DataCollector:
    def __init__(self):
        self.results = []

    def log(self, trial, iteration, system):
        self.results.append({
            "trial": trial,
            "iteration": iteration,
            "system": system
        })

    def summarize_results(self):
        self.results = sorted(self.results, key=lambda x: x["trial"])
        return 0

    def save_to_json(self, filename="simulation_results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)

    def load_from_json(self, filename="simulation_results.json"):
        with open(filename, "r") as f:
            self.results = json.load(f)
