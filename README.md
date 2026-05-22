# Atlas to GerryChain
A lightweight Python bridge to convert MCMC districting plans from the Atlas format into GerryChain Partition objects for downstream analysis.

## The Origin Story
This project was born out of a standard research bottleneck: bridging two different computational ecosystems to get a single diagnostic pipeline working.

Evaluating convergence and ensemble diversity on massive MCMC runs (e.g., long chains using `CycleWalk.jl`) requires calculating map-to-map distances, such as the Unconstrained Transfer Distance or Optimal Transport metrics, to run Partition-based Approximation for Convergence Evaluation (PACE) diagnostics, a recently developed method measuring chain convergence that relies on a good definition of distance between maps.

The broader redistricting software ecosystem already has excellent tools for calculating these distances, such as `wasserplan`, but they natively expect data wrapped in MGGG's `GerryChain` Partition objects. Conversely, high-performance Julia samplers frequently export states via the `Atlasio.jl` specification.

Instead of rewriting the underlying math for Optimal Transport or Variation of Information to accept `.jsonl` Atlas files, this repository serves as the "glue." It efficiently parses Atlas outputs and constructs `GerryChain` partitions on the fly, allowing researchers to plug high-performance Julia outputs directly into existing Python diagnostic tools. With this bridge, we can both celebrate the high-performance samplers based on `CycleWalk.jl` and the well-developed ecosystem surrounding `GerryChain`. 

## What It Does

- Reads the underlying state geography (e.g., a .json precinct graph) into networkx only once.
- Maps the stringified Atlas keys (like ["CINCINNATI 23-N_HAMILTON"]) to the internal integer node IDs used by GerryChain.
- Iterates through thinned .jsonl Atlas chains and translates the assignment dictionaries into valid Partition objects.
- Feeds the converted partitions into distance calculators to generate matrices for MCMC convergence analysis.
