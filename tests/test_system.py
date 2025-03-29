from app.models.agent import Agent
from app.models.resource import Resource
from app.models.system import System
from app.utils.constants import MC_UTILITY


def test_system_initialization():
    resources = [Resource(i, value=10) for i in range(5)]

    # Create mock agents with set actions
    action_set_1 = {frozenset([resources[0], resources[1]])}
    action_set_2 = {frozenset([resources[2], resources[3]])}
    agents = [
        Agent(agent_id=1, action_set=action_set_1, utility=MC_UTILITY),
        Agent(agent_id=2, action_set=action_set_2, utility=MC_UTILITY)
    ]

    system = System(resources=resources, agents=agents, m=2)

    assert len(system.resources) == 5
    assert len(system.agents) == 2
    assert system.M == 2


def test_system_score_default():
    resources = [Resource(i, value=10) for i in range(3)]

    # Create mock agents with set actions
    action_set_1 = {frozenset([resources[0], resources[1]])}
    action_set_2 = {frozenset([resources[1], resources[2]])}
    agents = [
        Agent(agent_id=1, action_set=action_set_1, utility=MC_UTILITY),
        Agent(agent_id=2, action_set=action_set_2, utility=MC_UTILITY)
    ]

    agents[0].current_action = set(resources[:2])  # Agent 1 covers resource 0 and 1
    agents[1].current_action = set(resources[1:])  # Agent 2 covers resource 1 and 2

    system = System(resources=resources, agents=agents, m=2)

    # System score should only count resource 1 (covered by both agents)
    assert system.system_score() == 10


def test_system_score_m_3():
    resources = [Resource(i, value=10) for i in range(3)]

    # Create mock agents with set actions
    action_set_1 = {frozenset([resources[0], resources[1]])}
    action_set_2 = {frozenset([resources[1], resources[2]])}
    action_set_3 = {frozenset([resources[0], resources[1], resources[2]])}
    agents = [
        Agent(agent_id=1, action_set=action_set_1, utility=MC_UTILITY),
        Agent(agent_id=2, action_set=action_set_2, utility=MC_UTILITY),
        Agent(agent_id=3, action_set=action_set_3, utility=MC_UTILITY)
    ]

    agents[0].current_action = set(resources[:2])  # Covers 0 and 1
    agents[1].current_action = set(resources[1:])  # Covers 1 and 2
    agents[2].current_action = set(resources)  # Covers 0, 1, and 2

    system = System(resources=resources, agents=agents, m=3)

    assert system.system_score() == 10


def test_system_score_no_agents():
    resources = [Resource(i, value=10) for i in range(3)]

    # Initialize a system with no agents
    system = System(resources=resources, agents=[], m=1)

    # No properties are covered, so score should be 0
    assert system.system_score() == 0


def test_system_score_no_resources():
    action_set = {frozenset()}
    agents = [Agent(agent_id=i, action_set=action_set, utility=MC_UTILITY) for i in range(2)]

    # Initialize a system with no properties
    system = System(resources=[], agents=agents, m=1)

    # No properties exist, so score should be 0
    assert system.system_score() == 0
