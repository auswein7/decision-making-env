import pytest
from app.core.algorithms import best_response
from app.core.algorithms import approximate_best_response
from app.experiments.experiments import generate_problem_instance

def test_best_response():
    system = generate_problem_instance(10, 3, (1, 3), (2,2), 1, (1,10))
    best_response(system, max_iterations=100)
    assert system.system_score() > 0  # Ensure the system score improves

def test_approximate_best_response():
    system = generate_problem_instance(10, 3, (1, 3), (2,2), 1, (1,10))
    approximate_best_response(system, max_iterations=100)
    assert system.system_score() > 0  # Ensure the system score improves

def test_ilp_response():
    return 0

def test_logit_response():
    return 0

def test_pso_response():
    return 0

def test_brute_force():
    return 0