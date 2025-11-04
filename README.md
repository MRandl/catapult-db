# Catapult

Yet another fast approximate nearest neighbor (ANN) search library in Rust.

## Features

- **Graph-based indexing**: Adjacency graph structure for ANN search using a mutable graph
- **High-performance vector operations**: L2 distance, dot product, normalization
- **Thread-safe indexing**: Concurrent indexing and search operations

## Requirements

- Rust nightly (uses `#![feature(portable_simd)]`)
- Vectors must be multiples of 8 elements (SIMD lane count)

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
# Run timing benchmark
cargo run --release

# Generate flamegraph
cargo flamegraph --profile release_symbols --bin catapult
```

## Project Structure

- `src/numerics/` - SIMD vector operations (L2, dot product, normalization)
- `src/indexing/` - Graph structures and beam search algorithms
- `src/candidates/` - Priority queues and candidate tracking for search
- `src/statistics/` - Statistical utilities

## License

See LICENSE file for details.
