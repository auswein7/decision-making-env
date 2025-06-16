import math
import statistics
from app.models.resource import Resource
from app.models.agent import Agent
from app.models.system import System

from app.utils.common_utils import (
    compute_resource_value_heterogeneity,
    compute_action_combinations,
    compute_agent_action_heterogeneity,
    compute_overlap_density,
    compute_agent_resource_entropy,
    compute_feasibility_margin
)

def generate_sample_system():
    """
    Create a fixed system with known structure for testing difficulty metrics.
    """
    resources = [
        Resource(0, 10),
        Resource(1, 20),
        Resource(2, 30),
        Resource(3, 40)
    ]

    agent_0 = Agent(
        agent_id=0,
        action_set=[
            {resources[0], resources[1]},
            {resources[2]}
        ],
        utility=None
    )

    agent_1 = Agent(
        agent_id=1,
        action_set=[
            {resources[1], resources[2]},
            {resources[3]}
        ],
        utility=None
    )

    agents = [agent_0, agent_1]
    M = 2
    system = System(resources=resources, agents=agents, m=M, id="test-system")

    return system

def test_difficulty_metrics():
    system = generate_sample_system()

    rh = compute_resource_value_heterogeneity(system.resources)
    ac = compute_action_combinations(system.agents)
    ah = compute_agent_action_heterogeneity(system.agents)
    od = compute_overlap_density(system.agents, system.resources)
    re = compute_agent_resource_entropy(system.agents, system.resources)
    fm = compute_feasibility_margin(system.agents, system.resources, system.M)

    print(f"Resource Heterogeneity: {rh:.4f}")
    print(f"Action Combinations: {ac}")
    print(f"Action Heterogeneity: {ah:.4f}")
    print(f"Overlap Density: {od:.4f}")
    print(f"Resource Entropy: {re:.4f}")
    print(f"Feasibility Margin: {fm:.4f}")

    assert math.isclose(rh, statistics.pstdev([10, 20, 30, 40]) / statistics.mean([10, 20, 30, 40]), rel_tol=1e-5)
    assert ac == 4  # 2 actions per agent → 2 × 2 = 4 joint combos
    assert ah == 0.0  # agent actions are uniform in size, and resources per action
    assert 0 <= od <= 1 #
    assert re > 1.0 # good resource dispersion in action sets
    assert fm < 0 # low fm system for this test
