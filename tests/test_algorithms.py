import pytest
from app.core.algorithms import best_response, approximate_best_response, logit_response, genetic_response, \
    ilp_response, brute_force, calculate_system_convergence

from app.experiments.experiments import generate_problem_instance


def test_best_response():
    system = generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0]
    best_response(system, max_iterations=100)
    assert system.system_score() > 0  # Ensure the system score improves


def test_approximate_best_response():
    system = generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0]
    approximate_best_response(system, max_iterations=100)
    assert system.system_score() > 0  # Ensure the system score improves


def test_logit_response():
    system = generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0]
    logit_response(system, max_iterations=100)
    assert system.system_score() > 0  # Ensure the system score improves


def test_genetic_response():
    return 0


# TODO:: TEST WILL FAIL UNTIL ILP FUNC IS FIXED
def test_ilp_response():
    system = generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0]
    assert brute_force(system)[0] == ilp_response(system)


def test_calculate_system_convergence():
    converge_iter = 10
    sys_history = [generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0] for _ in range(converge_iter)]
    for sys in sys_history:
        sys.score = 1
    current_sys = generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0]
    current_sys.score = 1
    converged = calculate_system_convergence(sys_history, current_sys)

    current_sys.score = 2
    not_converged = calculate_system_convergence(sys_history, current_sys)

    assert converged == True and not_converged == False
