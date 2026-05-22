"""
Core conversion logic for bridging Atlas-format MCMC outputs to GerryChain Partitions.
"""

import logging
from typing import Dict, Any
from gerrychain import Graph, Partition

# Configure basic logging for the module
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_base_graph(filepath: str, name_col: str = "NAME") -> tuple[Graph, Dict[str, int]]:
    """
    Loads a JSON precinct graph into GerryChain and builds a node ID crosswalk.
    
    By caching the crosswalk dictionary once, we avoid having to iterate over 
    the entire networkx graph for every single map in a 10-million step chain.

    Args:
        filepath (str): Path to the base graph JSON (e.g., OHpct20.json).
        name_col (str): The node attribute corresponding to the stringified keys 
                        in the Atlas output. Defaults to "NAME".

    Returns:
        tuple: A tuple containing:
            - The loaded GerryChain Graph object.
            - A crosswalk dictionary mapping the string names to internal integer IDs.
    """
    logger.info(f"Loading base graph from {filepath}...")
    try:
        graph = Graph.from_json(filepath)
    except Exception as e:
        logger.error(f"Failed to load graph from {filepath}. Error: {e}")
        raise

    crosswalk = {}
    missing_name_count = 0

    for nx_node_id, data in graph.nodes(data=True):
        if name_col in data:
            crosswalk[data[name_col]] = nx_node_id
        else:
            missing_name_count += 1

    if missing_name_count > 0:
        logger.warning(f"Found {missing_name_count} nodes missing the '{name_col}' attribute.")

    logger.info(f"Successfully mapped {len(crosswalk)} nodes.")
    return graph, crosswalk


def atlas_to_partition(atlas_map: Any, base_graph: Graph, crosswalk: Dict[str, int]) -> Partition:
    """
    Converts a single Atlas map object into a GerryChain Partition.

    Args:
        atlas_map (Any): An Atlas map object yielded by Atlas.nextMap().
        base_graph (Graph): The GerryChain Graph object loaded via load_base_graph.
        crosswalk (Dict[str, int]): The cached name-to-integer mapping.

    Returns:
        Partition: A GerryChain Partition object ready for metric evaluation.
    """
    assignment = {}
    missing_nodes = []

    # atlas_map.districting contains the assignments.
    # Keys often look like '["CINCINNATI 23-N_HAMILTON"]' due to Julia/Atlas serialization.
    for atlas_key, dist_id in atlas_map.districting.items():
        # Strip the brackets and quotes to isolate the raw string name
        clean_name = atlas_key.strip('[]"\'')
        
        if clean_name in crosswalk:
            nx_node_id = crosswalk[clean_name]
            assignment[nx_node_id] = dist_id
        else:
            missing_nodes.append(clean_name)

    if missing_nodes:
        # We only log the first 5 to prevent console spam on massively misaligned graphs
        logger.warning(
            f"Could not find {len(missing_nodes)} Atlas keys in the base graph. "
            f"Examples: {missing_nodes[:5]}"
        )

    return Partition(base_graph, assignment)