import uuid

import networkx as nx

from app.models.agent import Agent
from app.models.system import System
from app.models.resource import Resource
import random
import math

from app.utils.common_utils import export_scenario_to_json


def generate_graphs(args):
    funcs = {
        "erdos_renyi": generate_erdos_renyi_systems,
        "random_geometric": generate_random_geometric_systems,
        "watts_strogatz": generate_watts_strogatz_systems,
    }

    systems = funcs[args.graph_type](args)
    for sys in systems:
        export_scenario_to_json(system=sys)


def generate_erdos_renyi_systems(args):
    systems = []
    for _ in range(args.num_graphs):

        resources = [
            Resource(i, random.randint(args.resource_val_lb, args.resource_val_ub))
            for i in range(args.num_resources)
        ]

        # built in support for this graph class
        G = nx.bipartite.random_graph(
            args.num_agents,
            args.num_resources,
            args.erdos_prob,
            seed=None
        )

        agents = []
        for a in range(args.num_agents):
            neighbors = [
                resources[node - args.num_agents]
                for node in G.neighbors(a)
            ]
            agents.append(build_agent(a, neighbors, args))
        systems.append(System(resources, [a for a in agents if a is not None], args.max_cover, uuid.uuid4().hex))
    return systems


def generate_random_geometric_systems(args):
    """
    Place agents and resources in [x[0,1] y[0,1]]; edge if distance ≤ geo_radius.
    """
    systems = []
    for _ in range(args.num_graphs):
        # build resources and sample positions
        resources = [
            Resource(i, random.randint(args.resource_val_lb, args.resource_val_ub))
            for i in range(args.num_resources)
        ]
        pos_res = {r.id: (random.random(), random.random()) for r in resources}
        pos_ag = {a: (random.random(), random.random()) for a in range(args.num_agents)}

        # for each agent, find geometric neighbors
        agents = []
        for a in range(args.num_agents):
            ax, ay = pos_ag[a]
            neighbors = []
            for r in resources:
                rx, ry = pos_res[r.id]
                if math.hypot(ax - rx, ay - ry) <= args.geo_radius:
                    neighbors.append(r)
            agents.append(build_agent(a, neighbors, args))

        systems.append(System(resources, [a for a in agents if a is not None], args.max_cover, uuid.uuid4().hex))
    return systems


def generate_watts_strogatz_systems(args):
    """
    Generates graphs with lattice structure, higher ws_beta leads to more random structure. Low vals enforce
    spacial locality.

    :param args:
    :return:
    """
    systems = []
    num_r = args.num_resources
    for _ in range(args.num_graphs):
        resources = [
            Resource(i, random.randint(args.resource_val_lb, args.resource_val_ub))
            for i in range(num_r)
        ]

        # agent a -> resources (a + offset) % num_r
        base_edges = []
        k = min(args.ws_k, num_r - 1)
        half_k = k // 2
        for a in range(args.num_agents):
            for offset in range(-half_k, half_k + 1):
                if offset == 0: continue
                r_id = (a + offset) % num_r
                base_edges.append((a, r_id))

        # rewire edges relative to ws_beta
        edges = []
        for (a, r_id) in base_edges:
            if random.random() < args.ws_beta:
                # pick a new random resource
                r_id = random.randrange(num_r)
            edges.append((a, r_id))

        # create agents action sets
        agents = []
        for a in range(args.num_agents):
            neighbors = [
                resources[r_id]
                for (u, r_id) in edges if u == a
            ]
            agents.append(build_agent(a, neighbors, args))

        systems.append(System(resources, [a for a in agents if a is not None], args.max_cover, uuid.uuid4().hex))
    return systems


def build_agent(agent_id, neighbors, args):
    num_actions = len(neighbors) // 4
    action_set = []
    for _ in range(num_actions):
        subset = {r for r in neighbors if random.random() < args.coverage_probability}
        if subset:
            action_set.append(subset)
    return Agent(agent_id, action_set, utility=None) if len(action_set) > 0 else None
