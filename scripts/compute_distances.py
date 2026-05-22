#!/usr/bin/env python3
"""
Executable script for computing a pairwise distance matrix across an MCMC ensemble.
Designed for headless execution on compute grids to support PACE diagnostics.
"""

import argparse
import logging
import sys
import os
import numpy as np

# Explicitly add the project root to the Python path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import AtlasIO as Atlas
from src.converter import load_base_graph, atlas_to_partition
from src.metrics import unconstrained_transfer_distance, optimal_transport_distance

# Configure logging for grid output
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Compute pairwise distance matrices for Atlas MCMC chains.")
    parser.add_argument("--atlas", type=str, required=True, help="Path to the input .jsonl Atlas file.")
    parser.add_argument("--graph", type=str, required=True, help="Path to the base graph .json file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the resulting numpy matrix (.npy).")
    parser.add_argument("--metric", type=str, choices=["transfer", "wasserstein"], default="transfer",
                        help="Distance metric to compute (transfer or wasserstein).")
    parser.add_argument("--pop_col", type=str, default="POP20", help="Population column name for weighting.")
    parser.add_argument("--thin", type=int, default=10, help="Thinning interval. Process every Nth map.")
    
    args = parser.parse_args()

    # 1. Load the underlying geography
    base_graph, crosswalk = load_base_graph(args.graph, name_col="NAME")

    # 2. Extract and thin the MCMC chain
    logger.info(f"Opening Atlas file: {args.atlas}")
    logger.info(f"Extracting maps with a thinning interval of {args.thin}...")
    
    atlas_reader = Atlas.openAtlas(args.atlas)
    partitions = []
    step_count = 0
    
    while True:
        try:
            current_map = Atlas.nextMap(atlas_reader)
            # Atlas reader usually returns None or breaks when EOF is reached
            if current_map is None or not hasattr(current_map, 'districting'):
                break
                
            if step_count % args.thin == 0:
                part = atlas_to_partition(current_map, base_graph, crosswalk)
                partitions.append(part)
                
            step_count += 1
        except EOFError:
            break
        except Exception as e:
            logger.warning(f"Stopped reading Atlas file at step {step_count} due to: {e}")
            break

    Atlas.closeAtlas(atlas_reader)
    num_maps = len(partitions)
    logger.info(f"Successfully extracted {num_maps} maps for distance calculation.")

    if num_maps < 2:
        logger.error("Not enough maps extracted to compute a distance matrix. Decrease the thinning interval.")
        sys.exit(1)

    # 3. Compute the pairwise distance matrix
    logger.info(f"Computing the {args.metric} distance matrix ({num_maps}x{num_maps})...")
    dist_matrix = np.zeros((num_maps, num_maps))
    total_pairs = (num_maps * (num_maps - 1)) // 2
    pairs_computed = 0

    for i in range(num_maps):
        for j in range(i + 1, num_maps):
            if args.metric == "transfer":
                d = unconstrained_transfer_distance(partitions[i], partitions[j], weight_col=None)
            elif args.metric == "wasserstein":
                d = optimal_transport_distance(partitions[i], partitions[j], pop_col=args.pop_col)
            
            # Matrix is symmetric
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
            pairs_computed += 1
            if pairs_computed % max(1, total_pairs // 10) == 0:
                logger.info(f"Progress: {pairs_computed}/{total_pairs} pairs computed ({(pairs_computed/total_pairs)*100:.1f}%)")

    # 4. Save the results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, dist_matrix)
    logger.info(f"Distance matrix successfully saved to {args.output}")

if __name__ == "__main__":
    main()