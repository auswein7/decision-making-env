import uuid
from app.utils.constants import *
import networkx as nx

from app.models.agent import Agent
from app.models.system import System
from app.models.resource import Resource
import random
import math
import numpy as np

from app.utils.common_utils import export_scenario_to_json, calc_and_time_optimal, compute_system_difficulties

def generate_graphs(args):
    funcs = {
        ERDOS_RENYI: generate_erdos_renyi_systems,
        RAND_GEO: generate_random_geometric_systems,
        WATTS_STROGATZ: generate_watts_strogatz_systems,
    }

    params = []
    if args.param_sweep:
        params = compute_param_sweep_vals(args)

    if not params:
        if args.graph_type == ERDOS_RENYI:
            params = [args.erdos_prob]
        elif args.graph_type == RAND_GEO:
            params = [args.geo_radius]
        elif args.graph_type == WATTS_STROGATZ:
            params = [args.ws_beta]
        else:
            params = []

    for param in params:
        systems = funcs[args.graph_type](args, param)
        for sys in systems:
            sys.optimal_score, sys.optimal_coverage = calc_and_time_optimal(system=sys, init_from_opt=False)
            sys.feasibility_margin, sys.resource_entropy, sys.overlap_density, sys.agent_heterogeneity = compute_system_difficulties(systems=[sys])
            sys.generation_data = {"method": "graph", "graph_type": args.graph_type, "param": param}
            filter_unreachable_resources(system=sys)
            export_scenario_to_json(system=sys)

def compute_param_sweep_vals(args):
    if args.graph_type == ERDOS_RENYI:
        erdos_probs = np.arange(args.erdos_prob, MAX_ERDOS_PROB + 0.05/2, 0.05)
        return erdos_probs
    if args.graph_type == RAND_GEO:
        geo_radii = np.arange(args.geo_radius, MAX_GEO_RADIUS + 0.05 / 2, 0.05)
        return geo_radii
    if args.graph_type == WATTS_STROGATZ:
        ws_beta_vals = np.arange(args.ws_beta, MAX_WS_BETA + 0.05 / 2, 0.05)
        return ws_beta_vals
        
def filter_unreachable_resources(system):
    reachable = set()
    for agent in system.agents:
        for action in agent.action_set:
            reachable.update(action)

    system.resources = [
        res for res in system.resources
        if res in reachable
    ]

def generate_erdos_renyi_systems(args, erdos_prob):
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
            erdos_prob,
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


def generate_random_geometric_systems(args, geo_radius):
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
                if math.hypot(ax - rx, ay - ry) <= geo_radius:
                    neighbors.append(r)
            agents.append(build_agent(a, neighbors, args))

        systems.append(System(resources, [a for a in agents if a is not None], args.max_cover, uuid.uuid4().hex))
    return systems


def generate_watts_strogatz_systems(args, ws_beta):
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
            if random.random() < ws_beta:
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
