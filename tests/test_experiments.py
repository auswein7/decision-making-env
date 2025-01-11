import pytest
from app.experiments.experiments import generate_problem_instance, run_experiments

def test_generate_problem_instance():
    system = generate_problem_instance(10, 3, (1, 3), 2)
    assert len(system.resources) == 10
    assert len(system.agents) == 3

def test_run_experiments():
    run_experiments(3, 10, 3, 2)
