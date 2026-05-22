import sys
import os

# Explicitly add the project root to the Python path so it can find the 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import networkx as nx
import pytest
from gerrychain import Partition

# Now these imports will work natively
from src import AtlasIO as Atlas
from src.converter import atlas_to_partition

# Paths to the test data we just downloaded
TEST_ATLAS_PATH = "tests/test_data/atlas_truncated_nc_multiscale.jsonl"

@pytest.fixture
def base_graph_and_crosswalk():
    """
    Since we don't have the canonical NC graph JSON, we build a dummy graph 
    on the fly that contains the exact node names found in Jonathan's test atlas.
    """
    assert os.path.exists(TEST_ATLAS_PATH), f"Please download {TEST_ATLAS_PATH} first."
    
    # 1. Open the Atlas to peek at the keys
    atlas = Atlas.openAtlas(TEST_ATLAS_PATH)
    first_map = Atlas.nextMap(atlas)
    
    # 2. Build a dummy networkx graph using his exact keys
    dummy_graph = nx.Graph()
    crosswalk = {}
    
    # Jonathan's stringified keys usually look like '["VTD_123"]'
    for internal_id, (atlas_key, _) in enumerate(first_map.districting.items()):
        # Mimic the strip logic in converter.py to get the raw name
        clean_name = atlas_key.strip('[]"\'')
        
        # Add to networkx and our crosswalk
        dummy_graph.add_node(internal_id, NAME=clean_name)
        crosswalk[clean_name] = internal_id
        
    Atlas.closeAtlas(atlas)
    
    return dummy_graph, crosswalk

def test_atlas_to_partition_alignment(base_graph_and_crosswalk):
    """
    Tests that the converter successfully aligns Jonathan's Atlas format 
    into a GerryChain Partition without dropping any nodes.
    """
    base_graph, crosswalk = base_graph_and_crosswalk
    
    # Open the atlas again to grab the first map for the actual test
    atlas = Atlas.openAtlas(TEST_ATLAS_PATH)
    first_map = Atlas.nextMap(atlas)
    
    # --- Execute the Bridge ---
    partition = atlas_to_partition(first_map, base_graph, crosswalk)
    
    # --- Assertions ---
    # 1. Ensure it returned a valid GerryChain Partition
    assert isinstance(partition, Partition)
    
    # 2. Ensure every single node in the base graph was assigned a district
    assert len(partition.assignment) == len(base_graph.nodes)
    
    # 3. Verify a specific assignment matches what the Atlas said
    # Grab the first raw key and its district from Jonathan's map
    sample_raw_key = list(first_map.districting.keys())[0]
    sample_district = first_map.districting[sample_raw_key]
    
    # Translate it the way the converter does
    clean_name = sample_raw_key.strip('[]"\'')
    nx_node_id = crosswalk[clean_name]
    
    # The GerryChain partition should have assigned that exact district to the node ID
    assert partition.assignment[nx_node_id] == sample_district
    
    Atlas.closeAtlas(atlas)