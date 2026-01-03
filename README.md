# UseAIrs

## Important notes

1. **A big thanks to the Team of [CubeCL](https://github.com/Use-AIrs/cubecl). Make sure to check out their [Website](https://burn.dev/).**
2. **Preview Release** - Version 0.0.1: The release is planned this quarter. This is for the Government to validate that it actually works.
3. An AI framework in Rust for building parallel, locally hostable reinforcement learning systems in production environments.
4. This framework has been tested exclusively with CUDA 13. Other CUDA versions remain unverified and unsupported at this time. Rust 1.85 or later required.

## Code Philosophy

UseAIrs prioritizes explicit composition over implicit configuration. Rather than hiding complexity behind abstractions, we expose computational layers explicitly while maintaining type safety and zero cost abstractions.

The architecture separates concerns through clear interfaces. Data processing occurs independently from GPU computation. Models remain decoupled from training algorithms. Storage operates synchronously with explicit error handling. This separation allows replacing any component without modifying others.

Type safety extends to GPU operations through CubeCL. Device memory is tracked through RAII types. Kernel launches undergo compile time validation. Data races between GPU operations are prevented through explicit synchronization primitives. These guarantees eliminate entire classes of bugs common in CUDA programming while maintaining competitive performance.

For high abstraction use configuration drives behavior through JSON rather than code. Users define data sources, model architectures, and training procedures declaratively. The CLI validates configurations before execution, catching type errors and resource conflicts at startup rather than mid training.

## Example Use (dummy)

Load and execute via CLI:

```bash
cd crates/use-ai
cargo run --release
# Select "Config" -> Load configuration
# Select "Start" -> Execute training
```

Create a JSON configuration defining your workflow:

```json
{
  "name": "first_model",
  "version": "0.11_pre_alpha",
  "data": {
    "source": {
      "type": "csv",
      "path": "../../test.csv"
    },
    "scheme": {
      "columns": ["id", "age", "salary", "department", "city", "score", "status"]
    },
    "transformer": [
      {
        "t_id": 0,
        "operation": "categories",
        "params": {
          "columns": ["department", "status"]
        }
      }
    ]
  },
  "models": [
    {
      "id": 0,
      "model_type": "GradientBoostedDecisionTree",
      "input_columns": ["header1", "header2", "header3"],
      "target_columns": ["label"],
      "hyperparams": {
        "n_trees": 100,
        "learning_rate": 0.9,
        "max_depth": 5
      },
      "mode": "Train"
    }
  ],
  "output": {
    "final_output": ["m0"]
  }
}
```

Or programmatically:

```rust
use lib_store::{config_from_file, activate_config, get_active_config};
use lib_stage::stager;
use lib_calculator::{Tensor, Model};

// Load configuration
config_from_file("config.json".to_string(), "my_model".to_string())?;
activate_config("my_model".to_string())?;

// Process data
let config = get_active_config()?;
stager()?;

// Train model (implementation specific)
```

## Structure (WIP)

```
UseAIrs/
├── crates/
│   ├── core/
│   │   ├── lib-calculator/    # GPU computation with CubeCL
│   │   │   ├── tensor.rs      # Tensor type with shape/stride metadata
│   │   │   ├── model/         # Model trait + xgboost implementation
│   │   │   └── operation/     # GPU kernels (gpu/) + resource mgmt (mgmt/)
│   │   ├── lib-stage/         # CPU data processing with Rayon
│   │   │   ├── data/          # CSV/table parsers
│   │   │   └── output_guard.rs# Type safety for tensor construction
│   │   └── lib-store/         # MongoDB persistence (sync)
│   │       └── cfg/           # Configuration management
│   ├── use-ai/                # CLI with interactive menus
│   └── examples/
│       └── operation/         # NCCL, sorting, CSV benchmarks
└── docs/                      # mdBook documentation
```

**lib-calculator** contains all GPU-accelerated computation. Built on CubeCL, it provides the `Tensor<N>` type for device memory management alongside shape and stride tracking. The `Model` trait defines `execute(ctx, input) -> Output` for algorithm-agnostic model composition. The `xgboost` module implements gradient boosted decision trees with CPU tree construction and GPU gradient computation. The `operation` layer houses GPU kernels in `gpu/` and resource management in `mgmt/`.

**lib-stage** handles data transformation from raw formats to computation-ready tensors. Accepts CSV, JSON, or raw tables. Applies transformations like categorical encoding via Rayon parallelism. Constructs batches with correct memory layouts for GPU transfer. The `output_guard` validates tensor shapes and types before expensive operations begin.

**lib-store** manages persistence through MongoDB. Synchronous interface for configuration storage, training data metadata, and model checkpoints. The `cfg` module handles calculator, stage, and runtime configurations with versioning support.

**use-ai** provides the CLI interface. Interactive menus for configuration management, training execution, and inference. Validates JSON configurations before execution. Coordinates between store, stage, and calculator layers.

## Prerequisites

**Hardware**: NVIDIA GPU with CUDA compute capability 7.0+, minimum 8GB GPU memory

**Software**: CUDA 13, Rust 1.85+, MongoDB 6.0+

**Note**: Multi GPU training with NCCL requires the `nccl` feature flag.

## Quick Start

Build all components:

```bash
cargo build --release
```

Run benchmarks to verify installation:

```bash
cd crates/examples/operation
cargo run --release --bin sorting_bench
```

Launch CLI:

```bash
cd crates/use-ai
cargo run --release
```

## Current Limitations

Pre-alpha state. NCCL integration works only with CUDA 12.x. Multi-GPU training remains experimental. XGBoost supports only specific tree configurations. Error messages from GPU operations sometimes lack context.

The roadmap prioritizes expanded algorithm support for policy gradient methods, custom CubeCL kernels for complex RL operations, comprehensive testing suites, and production deployment tooling.

## Documentation (placeholder)

Full documentation will be available at `docs/`. Build with mdBook:

```bash
cd docs
mdbook serve
```

Navigate to `http://localhost:3000` to explore architecture details, API references, and implementation guides.

## License

See LICENSE.md for details.

## Funding

Was funded by Germany:

![Logo BMFTR](.assets/bmftr.jpg 'image title')
