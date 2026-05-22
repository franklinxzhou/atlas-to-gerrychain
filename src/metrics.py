"""
Metric wrappers for evaluating distances between GerryChain Partitions.
These are optimized for use in convergence diagnostics (e.g., PACE).
"""

import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from gerrychain import Partition

# Try to import wasserplan, but allow the script to gracefully degrade 
# if the user only wants the faster transfer distance.
try:
    from wasserplan import Pair
    HAS_WASSERPLAN = True
except ImportError:
    HAS_WASSERPLAN = False

logger = logging.getLogger(__name__)

def optimal_transport_distance(part_a: Partition, part_b: Partition, pop_col: str = "POP20") -> float:
    """
    Calculates the Wasserstein (Optimal Transport) distance between two plans.
    This solves a linear assignment problem to move population geographically.

    Args:
        part_a (Partition): The first map.
        part_b (Partition): The second map.
        pop_col (str): The node attribute containing the population. Defaults to "POP20".

    Returns:
        float: The exact transport distance (amount of population * distance moved).
    """
    if not HAS_WASSERPLAN:
        raise ImportError("The 'wasserplan' package is required for optimal transport distances.")
    
    # Pair initializes the CVXPY transport problem under the hood
    pair = Pair(part_a, part_b, indicator="population", pop_col=pop_col)
    
    return pair.distance


def unconstrained_transfer_distance(part_a: Partition, part_b: Partition, weight_col: str = None) -> float:
    """
    Calculates the minimum number of nodes (or population) that must change assignments 
    to match the two partitions, ignoring district label misalignment.
    
    This is formally known as the Minimum Hamming Distance up to permutation.

    Args:
        part_a (Partition): The first map.
        part_b (Partition): The second map.
        weight_col (str, optional): If provided (e.g., "POP20"), calculates the 
                                    population transfer distance. If None, calculates 
                                    the raw precinct-count transfer distance.

    Returns:
        float: The minimum number of transferred units (nodes or population).
    """
    # Ensure we are only comparing intersecting nodes
    nodes = set(part_a.graph.nodes) & set(part_b.graph.nodes)
    
    # Extract the unique district labels from both maps
    districts_a = list(set(part_a.assignment.values()))
    districts_b = list(set(part_b.assignment.values()))
    
    # Create internal indexing for the cost matrix
    idx_a = {d: i for i, d in enumerate(districts_a)}
    idx_b = {d: i for i, d in enumerate(districts_b)}
    
    # Initialize the cost matrix for the Hungarian algorithm.
    # We want to MAXIMIZE overlap, which means MINIMIZING negative overlap costs.
    cost_matrix = np.zeros((len(districts_a), len(districts_b)))
    total_weight = 0
    
    for node in nodes:
        d_a = part_a.assignment[node]
        d_b = part_b.assignment[node]
        
        # Determine the weight of this node
        weight = 1
        if weight_col:
            weight = part_a.graph.nodes[node].get(weight_col, 1)
            
        total_weight += weight
        
        # Subtract from the intersection matrix
        cost_matrix[idx_a[d_a], idx_b[d_b]] -= weight
        
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # The max overlap is the sum of the matched negative costs (inverted back to positive)
    max_overlap = -cost_matrix[row_ind, col_ind].sum()
    
    # The minimum transfer distance is simply the total weight minus the maximized overlap
    return total_weight - max_overlap