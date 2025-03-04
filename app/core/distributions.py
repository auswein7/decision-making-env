import random
import numpy as np

class Distribution:
    """
    Make this a parent class where all distributions inherit from.
    """

    def __init__(self, distribution, beta, temperature):
        print(f"Initializing distribution of type {distribution}")
        self.distribution = distribution
        self.beta = beta
        self.temperature = temperature

    def get_distribution(self):
        if self.distribution == "best_response":
            return self.best_response
        if self.distribution == "approximate_best_response":
            return self.approximate_best_response
        if self.distribution == "logit_response":
            return self.logit_response

    def best_response(self, action_scores):
        # Choose the best action deterministically
        max_score = max(action_scores.values())
        best_actions = [a for a in action_scores if action_scores[a] == max_score]
        best_action = random.choice(best_actions)
        return best_action

    def approximate_best_response(self, action_scores):
        max_score = max(action_scores.values())
        min_score = min(action_scores.values())

        # Create dict of [action -> new scaled score] based on passed beta value
        scaled_scores = {
            action: self.beta * (score - min_score) + (1 - self.beta) * (max_score - min_score)
            for action, score in action_scores.items()
        }

        # Select an action probabilistically based on scaled scores
        total_scaled_score = sum(scaled_scores.values())
        if total_scaled_score > 0:
            probabilities = {action: score / total_scaled_score for action, score in scaled_scores.items()}
            selected_action = random.choices(
                population=list(probabilities.keys()),
                weights=list(probabilities.values()),
                k=1
            )[0]
            return set(selected_action)
        else:
            return self.best_response(action_scores)

    def logit_response(self, action_scores):
        scores = np.array(list(action_scores.values()))
        max_score = np.max(scores)
        exp_scores = np.exp((scores - max_score) / self.temperature)
        sum_scaled_scores = np.sum(exp_scores)

        probabilities = {
            action: np.exp((score - max_score) / self.temperature) / sum_scaled_scores
            for action, score in action_scores.items()
        }

        return random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]
