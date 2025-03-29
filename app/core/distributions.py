import random

import numpy as np

from app.utils.constants import BEST_RESPONSE, APPROX_BEST_RESPONSE, LOGIT_RESPONSE


class Distribution:
    """
    This class defines distributions available to weight the agent actions.

    Attributes:
        distribution: requested distribution type (str)
        beta: beta value for approximate best response distribution
        temperature: temp value for logit response distribution
    """

    def __init__(self, distribution, beta, temperature):
        print(f"Initializing distribution of type {distribution}. Beta {beta}, Temperature {temperature}.")
        self.distribution = distribution
        self.beta = beta
        self.temperature = temperature

    def get_distribution(self):
        if self.distribution == BEST_RESPONSE:
            return self.best_response
        if self.distribution == APPROX_BEST_RESPONSE:
            return self.approximate_best_response
        if self.distribution == LOGIT_RESPONSE:
            return self.logit_response

    def best_response(self, action_scores):
        """
        Implements best response action selection strategy:

        Choose the action that maximizes the score deterministically, pure greedy.
        """
        max_score = max(action_scores.values())
        best_actions = [a for a in action_scores if action_scores[a] == max_score]
        best_action = random.choice(best_actions)
        return best_action

    def approximate_best_response(self, action_scores):
        """
        Implements approximate best response action selection strategy:

        Let:
            s_max = max score
            s_min = min score
            s_a = score of action a

        Scaled score for action a is:
            scaled_a = β * (s_a - s_min) + (1 - β) * (s_max - s_min)

        Where:
            - β ∈ [0, 1] controls the level of best-response approximation
                - β = 1 → exact best response
                - β < 1 → random behaviors
        """
        max_score = max(action_scores.values())
        min_score = min(action_scores.values())

        scaled_scores = {
            action: self.beta * (score - min_score) + (1 - self.beta) * (max_score - min_score)
            for action, score in action_scores.items()
        }

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
            return random.choice([a for a in action_scores])

    def logit_response(self, action_scores):
        """
        Implements the logit (softmax) response action selection strategy:

            Let:
                - s_a = score of action a
                - T = temperature parameter (T > 0)
                - s_max = max score among all actions
                - A = set of all available actions

        The probability of choosing action a ∈ A is:

            P(a) = exp((s_a - s_max) / T) / Σ_{b ∈ A} exp((s_b - s_max) / T)

        Where:
            - T → 0: exact best response
            - T → ∞: random behaviors
        """
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
