# Overview

## I: Introduction

Building artificial intelligence systems for practical production environments asks to address fundamental challenges in computational efficiency, model composition, and hardware utilization. Most frameworks optimize for either research or production, rarely both.

This introduces us to several problems that need to be addressed. Research demands flexibility in architecture and training procedures. Production requires deterministic behavior and reproducible results. Bridging these requirements within a single framework leads to trade-offs that most ecosystems resolve by fragmentation.

UseAIrs approaches this differently. Rather than hiding complexity behind abstractions, we expose computational layers explicitly while providing composition tools that maintain flexibility. This allows researchers to experiment with novel architectures while production engineers utilize identical implementations under strict resource bounds.

The following documentation explores the architectural decisions underlying UseAIrs. Each component addresses specific computational challenges inherent in building reinforcement learning systems for production deployment.

## II: Project Structure

UseAIrs organizes its codebase into distinct layers. Understanding this structure is essential for both users and contributors.

```
UseAIrs/
├── crates/
│   ├── examples/          # Benchmarks and examples
│   ├── use-ai/            # CLI binary
│   └── core/              # Core libraries
│       ├── lib-stage/     # Data processing
│       ├── lib-calculator/# GPU computation
│       └── lib-store/     # Persistence
├── docs/                  # Documentation
└── ibm_sample_data/       # Sample data
```

This documentation walks through each component iteratively. We start with examples that demonstrate usage patterns, move through the CLI, and descend into the core libraries.

# 1: Examples and Benchmarks

*Location: `crates/examples/operation`*

The examples crate provides practical demonstrations alongside performance benchmarking tools. These validate the framework's implementation and offer starting points for users.

## I: Available Benchmarks

**NCCL Benchmark** (`nccl_bench`) measures GPU memory bandwidth and inter-device communication. Compares Rust implementations using CubeCL against PyTorch's NCCL primitives. Provides detailed latency statistics: min, median, P95, P99, max. Results save to markdown files.

**Sorting Benchmark** (`sorting_bench`) tests GPU-accelerated sorting operations. Demonstrates efficient kernel execution while maintaining Rust's type safety and memory correctness.

**CSV Benchmark** (`csv_bench`) evaluates data loading and preprocessing pipelines. Stresses the stage component's ability to transform structured data into computation-ready tensors.

## II: Running Benchmarks

**IMPORTANT**: These benchmarks have been tested exclusively with CUDA 12.x. Other versions remain unverified.

### NCCL Benchmark

```bash
cd crates/examples/operation
cargo run --release --bin nccl_bench --features nccl
```

Interactive prompts for data size (4 MB to 4 GB) and iteration counts. Results include timing statistics, bandwidth measurements, and stability metrics like coefficient of variation and jitter.

### Sorting Benchmark

```bash
cargo run --release --bin sorting_bench
```

Generates large datasets, sorts on GPU, validates correctness while measuring throughput.

### Understanding Results

Timing statistics show minimum latency for best-case performance, median and average for expected behavior, P95/P99 for worst-case analysis. Quality indicators include coefficient of variation where values below 5% indicate excellent stability. Jitter below 10% represents low variance. Memory bandwidth shows both peak performance and average real-world throughput.

## III: Integration

Each benchmark demonstrates patterns documented in later sections. NCCL showcases GPU memory management from lib-calculator. CSV illustrates data pipelines from lib-stage. Sorting combines both layers for complete workflows.

Users should examine benchmark source alongside this documentation to understand practical implementation patterns. These represent production-quality examples rather than simplified demonstrations.

# 2: CLI Tool - use-ai

*Location: `crates/use-ai`*

The `use-ai` binary provides the framework's primary interface for users without deep Rust knowledge. It abstracts complexity behind JSON configuration and interactive menus.

## I: Architecture

**Main Entry** (`main.rs`) initializes the application, parses command-line arguments, coordinates between menu systems. Establishes runtime environment and ensures proper cleanup on exit.

**Start Menu** (`start_menu.rs`) presents the initial interface. Users select between configuration management, model training, inference execution, or benchmark runs. Handles input validation and dispatches to appropriate subsystems.

**Configuration Menu** (`config_menu.rs`) manages JSON configuration files. Users create new configurations, validate existing ones, or load configurations for execution. Provides interactive prompts guiding through required fields and valid value ranges.

**Error Handling** (`error.rs`) defines custom error types for CLI operations. Configuration parsing errors, file system errors, communication failures with core libraries.

## II: Configuration System

The CLI operates entirely through JSON configuration files. A typical configuration specifies data sources, model architecture, training procedures, and output locations.

The configuration menu validates all fields before execution. Type errors, missing fields, and invalid ranges are caught at validation time rather than during execution. Prevents wasted computation on malformed configurations.

## III: Execution Flow

When users select a training or inference task, the CLI loads and validates the JSON configuration. Initializes storage through lib-store. Constructs data processing pipeline via lib-stage. Instantiates models and schedules computation through lib-calculator. Monitors execution and reports progress.

This workflow demonstrates how the three core libraries integrate within a complete application. The CLI serves as reference implementation for users building custom workflows programmatically.

# 3: Data Processing - lib-stage

*Location: `crates/core/lib-stage`*

The stage library handles all data transformation from raw formats to computation-ready tensors. Bridges the gap between diverse data sources and strict type requirements of GPU computation.

## I: Core Responsibilities

Accepts structured data in JSON, CSV, or raw tables. Applies normalization, scaling, and encoding transformations. Constructs batches with appropriate shapes for model consumption. Produces output tensors compatible with lib-calculator.

This pipeline executes on CPU using parallel processing via Rayon. Leverages ndarray for efficient numeric operations while maintaining zero-copy semantics where possible.

## II: Module Organization

**Data Module** (`src/data/`) contains format-specific parsers and transformers. The `for_raw_table.rs` submodule handles tabular data structures common in reinforcement learning environments.

**Output Guard** (`output_guard.rs`) ensures type safety during tensor construction. Validates that output shapes match expected dimensions and data types align with downstream requirements. Catches dimension mismatches before expensive GPU operations begin.

**Error Types** (`error.rs`) defines stage-specific errors. Parsing failures, transformation errors, validation failures. These propagate to the CLI with actionable messages.

## III: Processing Pipeline

A typical pipeline proceeds through phases. Ingest reads raw data from configured sources. Transform applies normalization and feature engineering. Batch groups samples into appropriately sized collections. Output constructs tensors with correct memory layouts for GPU transfer.

Each phase executes with explicit error handling. Failures include detailed context about which transformation failed and why. This aids debugging when integrating new data sources.

## IV: Integration Points

Interfaces with lib-store to retrieve data source configurations. Interfaces with lib-calculator to determine required tensor formats. Otherwise operates independently, enabling users to test preprocessing pipelines without GPU resources.

# 4: GPU Computation - lib-calculator

*Location: `crates/core/lib-calculator`*

The calculator library contains all GPU-accelerated computation. Built on CubeCL, provides explicit control over device memory and kernel execution while exposing a safe Rust interface.

## I: Architecture Layers

**Tensor Layer** (`tensor.rs`) defines the fundamental data structure for GPU computation. Tensors encapsulate device memory alongside shape and stride information. Type system ensures compile-time verification of dimension compatibility. Supports both CPU and GPU representations with explicit transfer operations between memory spaces.

**Model Layer** (`src/model/`) implements machine learning algorithms. Provides trait definitions for models, configurations, and statistics. The `xgboost/` submodule contains gradient boosted decision tree implementations for tabular data. Additional model types integrate through the same trait system.

**Operation Layer** (`src/operation/`) houses GPU kernels and computational primitives. The `gpu/` submodule contains CubeCL kernel implementations for matrix multiplication, activation functions, and loss calculations. The `mgmt/` submodule handles resource allocation, kernel scheduling, and device synchronization.

## II: Model System

Models implement a minimal trait defining input and output types, inference configuration, and training configuration. This decoupling allows the same model to serve multiple algorithms and use cases.

The XGBoost implementation demonstrates this flexibility. Provides tree construction on CPU with gradient computation on GPU. Trains efficiently on tabular data while maintaining the same interface as neural network models. Users can substitute XGBoost for neural networks in reinforcement learning algorithms without modifying training code.

## III: Operation Management

The operation layer manages GPU resources explicitly. Tracks device memory allocations. Schedules kernel execution to maximize utilization. Synchronizes operations across multiple GPUs when available.

NCCL integration enables multi-GPU communication for distributed training. This remains optional behind a feature flag, allowing single-GPU deployments to avoid the dependency.

## IV: Safety Guarantees

Despite low-level GPU operations, the calculator maintains Rust's safety guarantees. Device memory is tracked through RAII types that ensure cleanup. Kernel launches undergo compile-time validation of argument types and counts. Data races between GPU operations are prevented through explicit synchronization primitives.

These guarantees eliminate entire classes of bugs common in CUDA programming. Memory leaks, use-after-free errors, and race conditions are caught at compile time or prevented by the type system.

## V: Performance

The calculator targets production performance requirements. Kernel implementations minimize memory transfers through careful data layout. Operations fuse where possible to reduce intermediate allocations. Resource pooling reduces allocation overhead during training loops.

NCCL benchmarks in the examples crate validate these optimizations. Results demonstrate competitive performance with PyTorch while maintaining stronger safety guarantees.

# 5: Persistence - lib-store

*Location: `crates/core/lib-store`*

The store library handles all data persistence through MongoDB. Manages configuration storage, training data access, and model checkpoint serialization.

## I: Design Philosophy

Rather than abstracting database operations behind generic traits, lib-store exposes MongoDB's document model directly. This prioritizes explicitness over portability. Users understand exactly where data resides and how it structures.

The synchronous interface simplifies reasoning about consistency. All database operations complete before returning control. This trades some throughput for predictable behavior.

## II: Module Organization

**Configuration Module** (`src/cfg/`) manages framework configurations. The `calc.rs` submodule handles calculator configurations including device selection and memory limits. The `stage.rs` submodule stores data processing pipeline definitions.

**Core Library** (`lib.rs`) provides database client and connection management. Handles connection pooling, authentication, and error recovery. Exposes collection interfaces for each data type stored.

## III: Storage Patterns

Organizes data into logical collections. Configurations reside in a dedicated collection with versioning support. Training datasets reference external storage with metadata in MongoDB. Model checkpoints serialize to binary format with metadata documents tracking versions and performance metrics.

This organization enables operational patterns. Users can roll back to previous configurations when experiments fail. Training runs reference immutable dataset versions ensuring reproducibility. Model registries track performance across versions enabling automated selection of best checkpoints.

## IV: Integration Points

The store library integrates with all other components. CLI loads configurations from storage at startup. Stage library queries dataset metadata to construct processing pipelines. Calculator library persists model checkpoints during training.

This central role makes the store library critical for production deployments. The synchronous interface and explicit error handling ensure that persistence failures surface immediately rather than corrupting state silently.

# 6: Putting It Together

The components integrate to form complete AI workflows. A typical reinforcement learning training run proceeds as follows:

CLI loads a JSON configuration from lib-store specifying the environment, algorithm, and hyperparameters. Stage library constructs a data processing pipeline based on the configuration, reading training samples and transforming them into tensors. Calculator library initializes the model on GPU and executes training loops, computing gradients and updating parameters. Periodically, the calculator serializes model checkpoints to lib-store. Upon completion, final metrics and the trained model persist to storage.

This workflow demonstrates how explicit composition replaces framework magic. Each component has clear responsibilities and interfaces. Users can replace any component with custom implementations without modifying other layers.

## I: Extension Points

New model architectures implement the model trait in lib-calculator. Custom data formats add parsers to lib-stage. Alternative storage backends can replace lib-store while maintaining the same interface. Additional GPU operations extend the operation layer with new kernels.

These extension points enable the framework to grow with user requirements. The core architecture remains stable while domain-specific functionality integrates through well-defined interfaces.

## II: Current Limitations

UseAIrs exists in alpha state. Several areas require hardening before production deployment:

NCCL integration works only with CUDA 12.x installations. Multi-GPU training remains experimental with limited testing. XGBoost implementation supports only specific tree configurations. Error messages from GPU operations sometimes lack context for debugging.

These represent known work areas rather than fundamental architectural flaws. The roadmap prioritizes addressing them through expanded testing, improved error handling, and documentation enhancements.

# 7: Getting Started

## I: Prerequisites

**Hardware**: NVIDIA GPU with CUDA compute capability 7.0 or higher. CUDA 12.x installation. Minimum 8GB GPU memory for typical workloads.

**Software**: Rust toolchain 1.70 or later. MongoDB 6.0 or later for persistence. CubeCL with CUDA backend enabled.

**IMPORTANT**: The framework has been tested exclusively with CUDA 12.x. Other versions remain unverified and unsupported at this time.

## II: Building from Source

Clone the repository and build all components:

```bash
git clone <repository-url>
cd UseAIrs
cargo build --release
```

This builds all crates including the CLI tool, core libraries, and examples. The build process validates that CUDA is properly configured and accessible.

## III: Running Examples

Start with the benchmarks to verify your installation:

```bash
cd crates/examples/operation
cargo run --release --bin sorting_bench
```

Successful execution confirms that GPU operations work correctly. The benchmark outputs performance metrics and validates results for correctness.

## IV: Using the CLI

Launch the interactive CLI:

```bash
cd crates/use-ai
cargo run --release
```

Follow the menu prompts to create a configuration, load sample data, and execute a training run. The CLI guides you through all required steps with validation at each stage.

## V: Next Steps

After validating your installation with examples and the CLI, explore the individual library documentation for programmatic usage. Each library includes API documentation with usage examples. The [`lib-calculator::model`](/calc/model.md) module provides a good entry point for understanding model composition patterns.

---

**Navigation**: Use the sidebar to explore detailed documentation for each component:

- **[`lib-store`](/store/store.md)**: Storage and persistence
- **[`lib-stage`](/stage/stage.md)**: Data processing pipelines
- **[`lib-calculator`](/calc/calc.md)**: GPU computation and models
- **[`lib-proc_macros`](/macros/macros.md)**: Compile-time code generation
- **[`use_ai`](/tooling/useai.md)**: CLI tool reference
