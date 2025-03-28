from app.runner.experiments import generate_problem_instance
from app.utils.common_utils import load_scenario_from_json
from app.utils.constants import JSON_LOAD_PATH


def test_generate_problem_instance():
    system = generate_problem_instance(10, 3, (1, 3), (2, 2), 1, (1, 10), 1)[0]
    assert len(system.resources) == 10
    assert len(system.agents) == 3
    assert system.M == 1


def test_generate_problem_instance_from_json():
    system = load_scenario_from_json("../" + JSON_LOAD_PATH)
    assert len(system.resources) == 10
    assert len(system.agents) == 5
    assert system.M == 3
    return 0
