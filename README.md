# CatapultDB 


Yet another fast approximate nearest neighbor (ANN) search library in Rust.
It is based on a proximity graph data structure that enables fast nearest neighbor search at the cost of being approximate.
We improve by quite a long shot over the current SoTA by inserting edges that 'remember' past trajectories in the 
graph data structure. These special history-based edges are nicknamed 'catapults'.

Requires Rust nightly (uses `#![feature(portable_simd)]`). Any cargo/Rust install will be able to handle this repo, possibly by downloading some extras for your compiler.

## Building

```bash
# Standard release build
cargo build --release

# With debug symbols (for profiling)
cargo build --profile release_symbols
```

## Benchmarking

The project includes a benchmark harness for profiling vector operations:

```bash
# Generate flamegraph
cargo flamegraph --profile release_symbols --bin catapult
```

## Project Structure
- `slides/` - some slides used for internal presentations at SaCS
- `src/numerics/` - SIMD vector operations (L2, dot product, normalization)
- `src/indexing/` - Graph structures and beam search algorithms
- `src/candidates/` - Priority queues and candidate tracking for search
- `src/statistics/` - Statistical utilities

## License

See LICENSE file for details.
