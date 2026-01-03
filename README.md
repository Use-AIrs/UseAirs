# UseAIrs

## Important notes

1. **A big thanks to the Team of [CubeCL](https://github.com/Use-AIrs/cubecl). Make sure to check out their [Website](https://burn.dev/).**
2. **Preview Release** Version 0.0.1: The release is planned this quarter. This is for the Government to validate that it actually works.
3. This framework has been tested exclusively with CUDA 13. Other CUDA versions remain unverified and unsupported at this time. Rust 1.85 or later required.
4. This is only a mirror, make sure to check out the [official repository](https://code.q-network.org/Q-Research/UseAIrs). Thanks to the Q-Network for Hosting the repository within European restrictions.
5. Shout out to the [Q-Network GmbH](https://q-network.org/)! Make sure to check out their [Q-Research](https://code.q-network.org/Q-Research) department who will release useful tooling every now and then exclusively within the Q-Network. 

## Code Philosophy

An AI framework in Rust for building parallel, locally hostable reinforcement learning systems in production environments.

This only works with the cubecl mock release. The version uploaded uses an old version of the NCCL module for CubeCl CUDA.
https://github.com/Use-AIrs/cubecl-mock
https://code.q-network.org/Q-External/CubeCl

When placing the CubeCl mock next to the UseAIrs preview everything should work out just fine.
In any case feel free to share problems within the issues tab.
Happy to see reports with different GPUs!

## Prerequisites

**Hardware**: NVIDIA GPU with CUDA compute capability 12.0+, minimum 8GB GPU memory

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


## Current Limitations

Pre-alpha state. NCCL integration works only with CUDA 13.x. Multi-GPU training remains experimental. XGBoost supports only specific tree configurations. Error messages from GPU operations sometimes lack context.

The roadmap prioritizes expanded algorithm support for policy gradient methods, custom CubeCL kernels for complex RL operations, comprehensive testing suites, and production deployment tooling.

## Documentation (placeholder)

Full documentation will be available at `docs/`. Build with mdBook:

```bash
cd docs
mdbook serve
```

Navigate to `http://localhost:3000` to explore architecture details, API references, and implementation guides.

## License

This project is licensed under the PolyForm Perimeter License 1.0.1.

## With Release:

### Free Use for Qualifying Organizations

Charitable organizations, educational institutions, public research 
organizations, public safety or health organizations, environmental 
protection organizations, and government institutions may apply for 
a free license.

**Application Process:**  
Organizations in the above categories can request a free license at any 
time. Upon verification of your organization's positive societal impact 
and background check, we will grant a complimentary license.

Please contact: admin@q-network.org
 
### Commercial Use

All commercial use requires a paid license.

**Evaluation licenses available** (14 - 28 days)

**Contact:** admin@q-network.org

## Funding

Was funded by Germany:

![Logo BMFTR](.assets/bmftr.jpg 'image title')

