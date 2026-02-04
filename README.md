# CatapultDB

Yet another fast approximate nearest neighbor (ANN) search library in Rust.
It is based on a proximity graph data structure that enables fast nearest neighbor search at the cost of being approximate.
We improve by quite a long shot over the current SoTA by inserting edges that 'remember' past trajectories in the 
graph data structure. These special history-based edges are nicknamed 'catapults'.

## Requirements

- Rust nightly (uses `#![feature(portable_simd)]`)
- Python and Maturin should be available on PATH if you intend to build the Python bindings.

## Building

```bash
# Standard release build
cargo build --release

# With debug symbols (for profiling)
cargo build --profile release_symbols
```

## Usage

```bash
cargo run --release -- \
  --queries <path/to/queries.npy> \
  --graph <path/to/graph_metadata> \
  --payload <path/to/graph_payload> \
  --num-neighbors <k> \
  --beam-width <width> \
  [--catapults] \
  [--threads <n>]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `-q, --queries` | Path to queries file (NumPy `.npy` format) |
| `-g, --graph` | Path to graph metadata file (binary format) |
| `-p, --payload` | Path to graph payload/vectors file |
| `--num-neighbors` | Number of nearest neighbors to return (k) |
| `--beam-width` | Beam search width (larger = more accurate but slower) |

### Optional Arguments

| Flag | Description |
|------|-------------|
| `-c, --catapults` | Enable catapult optimization |
| `-t, --threads` | Number of threads (default: 1) |

### Example

```bash
cargo run --release -- \
  -q test_index/queries.npy \
  -g test_index/graph.bin \
  -p test_index/payload.bin \
  --num-neighbors 10 \
  --beam-width 64 \
  --catapults \
  --threads 4
```

## Profiling

```bash
# Generate flamegraph (requires cargo-flamegraph)
cargo flamegraph --profile release_symbols --bin catapult
```

## Datasets

### File Formats

- **Queries**: NumPy `.npy` format containing float32 vectors
- **Graph metadata**: Custom binary format (adjacency lists)
- **Graph payload**: Custom binary format (vector data, 64-byte aligned)

## Project Structure

- `src/numerics/` - SIMD vector operations (L2 distance, dot product)
- `src/search/` - Graph structures and beam search algorithms
- `src/sets/` - Data structures for search (candidates, catapults, visited nodes)
- `src/fs/` - File I/O for graphs and NumPy vectors
- `src/statistics/` - Statistical utilities
- `slides/` - Internal presentation slides

## License

See LICENSE file for details.
