// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::base::*;
use std::marker::PhantomData;

pub trait AdaptiveKernel<R: Runtime, N: Numeric + CubeElement>: Kernel<R, N> {
    
    fn kernel_id(&self) -> &'static str;

    /// Estimate memory footprint in bytes
    /// Used to decide between SingleBlock/MultiBlock/Streaming strategies
    fn memory_footprint(&self, input_shapes: &[Vec<usize>]) -> usize {
        input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>() * std::mem::size_of::<N>())
            .sum()
    }

    /// Compute intensity: FLOPs per byte accessed
    /// Higher values (>10) = compute-bound, can skip some memory optimizations
    /// Lower values (<10) = memory-bound, focus on coalescing and shared memory
    fn compute_intensity(&self) -> f32 {
        1.0 // Default: memory-bound
    }

    /// Should this kernel use shared memory?
    /// From docs/Advanced_Patterns_Final.md: < 8KB benefits from shared memory
    fn benefits_from_shared(&self, total_size: usize) -> bool {
        total_size < 8192 && self.compute_intensity() < 10.0
    }

    /// Does this kernel require warp-level synchronization?
    fn requires_warp_sync(&self) -> bool {
        false
    }
}

/// Trait for kernels that support line-based vectorization
///
/// Vectorization loads/stores multiple elements per thread (typically 4 or 8)
/// See docs/GPU_Programming_Guide.md section "Vektorisierung & VirtualTensor"
///
/// Benefits:
/// - 4-8x memory bandwidth utilization
/// - Reduced number of memory transactions
/// - Better cache utilization
pub trait VectorizedKernel<R: Runtime, N: Numeric + CubeElement>: AdaptiveKernel<R, N> {
    /// Optimal line size for this data type
    /// Target: 16-byte aligned memory transactions
    ///
    /// f16: 8 elements * 2 bytes = 16 bytes
    /// f32: 4 elements * 4 bytes = 16 bytes
    /// f64: 2 elements * 8 bytes = 16 bytes
    fn optimal_line_size(&self) -> u32 {
        let dtype_size = std::mem::size_of::<N>();
        match dtype_size {
            2 => 8,  // f16
            4 => 4,  // f32
            8 => 2,  // f64
            _ => 1,
        }
    }

    /// Can this kernel efficiently use vectorization?
    fn supports_vectorization(&self) -> bool {
        true
    }

    /// Check if tensor size is aligned to line size
    fn is_line_aligned(&self, size: usize) -> bool {
        size % self.optimal_line_size() as usize == 0
    }

    /// Pad size to line alignment (for algorithms requiring power-of-2 like bitonic sort)
    fn pad_to_line_size(&self, size: usize) -> usize {
        let line_size = self.optimal_line_size() as usize;
        let remainder = size % line_size;
        if remainder == 0 {
            size
        } else {
            size + (line_size - remainder)
        }
    }
}

// ============================================================================
// MEMORY ACCESS PATTERNS
// ============================================================================

/// Memory access pattern classification
///
/// Critical for performance: coalesced (Parallel) access is ~10x faster than strided
/// See docs/Advanced_CubeCL_Book.md "Parallel vs. Perpendicular - Der entscheidende Unterschied"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Parallel: threads access contiguous memory (FAST)
    /// Example: reducing rows in row-major layout
    /// Thread 0: [a0, a1, a2, a3]
    /// Thread 1: [a4, a5, a6, a7]
    /// Memory transactions are coalesced!
    Parallel {
        /// Step size between thread accesses
        step_size: u32,
    },

    /// Perpendicular: threads access strided memory (SLOWER but optimizable with Lines)
    /// Example: reducing columns in row-major layout
    /// Thread 0: [a0, b0, c0, d0] (stride = row_width)
    /// Thread 1: [a1, b1, c1, d1]
    /// Still efficient with Line<T> vectorization!
    Perpendicular {
        /// Stride between elements
        stride: u32,
    },

    /// Random: no pattern (SLOWEST, avoid if possible)
    Random,
}

/// Trait for operations that work along a reduction axis
///
/// Key decision: Is the reduction axis parallel or perpendicular to memory layout?
/// This determines optimal memory access pattern and kernel configuration.
pub trait ReductionKernel<R: Runtime, N: Numeric + CubeElement>: VectorizedKernel<R, N> {
    /// Determine access pattern for reduction axis
    ///
    /// For row-major layout [M, N]:
    /// - axis=0 (reduce rows): Perpendicular (stride = N)
    /// - axis=1 (reduce cols): Parallel (contiguous)
    fn access_pattern(&self, axis: usize, shape: &[usize], strides: &[usize]) -> AccessPattern {
        let rank = shape.len();

        // Last axis in row-major = contiguous = Parallel
        if axis == rank - 1 {
            let line_size = self.optimal_line_size();
            AccessPattern::Parallel {
                step_size: line_size * 256, // 256 threads per block typical
            }
        } else {
            // Other axes = strided = Perpendicular
            let stride = strides[axis] / self.optimal_line_size() as usize;
            AccessPattern::Perpendicular {
                stride: stride as u32,
            }
        }
    }

    /// Calculate reduction stride for perpendicular access
    /// From docs/GPU_Programming_Guide.md "Perpendicular Mode"
    fn reduction_stride(&self, axis: usize, strides: &[usize]) -> u32 {
        let line_size = self.optimal_line_size();
        (strides[axis] / line_size as usize) as u32
    }

    /// Is reduction parallel to memory layout?
    fn is_parallel(&self, axis: usize, rank: usize) -> bool {
        axis == rank - 1
    }
}

// ============================================================================
// EXECUTION STRATEGIES
// ============================================================================

/// Execution strategy based on data size
///
/// Decision tree from docs/Advanced_Patterns_Final.md "Pattern Selection Decision Tree"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Single block with shared memory (< 8KB)
    /// Best for: Small reductions, histogram building
    /// Pattern: All data fits in shared memory, no cross-block communication
    SingleBlock {
        block_size: u32,
        shared_mem_bytes: usize,
    },

    /// Multiple blocks, optional shared memory (8KB - 1MB)
    /// Best for: Medium element-wise ops, reductions
    /// Pattern: Each block processes chunk, may use shared for local reductions
    MultiBlock {
        block_size: u32,
        num_blocks: u32,
        use_shared: bool,
    },

    /// Streaming with chunking (> 1MB)
    /// Best for: Large datasets, out-of-core processing
    /// Pattern: Process in chunks with double buffering
    /// See docs/Advanced_CubeCL_Book.md "Out-of-Core Processing Pattern"
    Streaming {
        chunk_size: usize,
        num_chunks: usize,
    },

    /// Multi-GPU distributed (> 100MB)
    /// Best for: Huge datasets, training large models
    /// Pattern: Data parallel with NCCL collectives
    /// See docs/Advanced_CubeCL_Book.md "Multi-GPU Coordination Patterns"
    Distributed {
        num_devices: usize,
        strategy: DistributionStrategy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionStrategy {
    /// Split batch dimension across GPUs
    DataParallel,

    /// Split tensor itself across GPUs
    TensorParallel,

    /// Split layers across GPUs (for sequential models)
    PipelineParallel,
}

/// Strategy selector that chooses optimal execution based on operation characteristics
pub struct StrategySelector {
    /// Thresholds for strategy transitions
    pub small_threshold: usize,      // 8KB default
    pub medium_threshold: usize,     // 1MB default
    pub large_threshold: usize,      // 100MB default
}

impl Default for StrategySelector {
    fn default() -> Self {
        Self {
            small_threshold: 8192,
            medium_threshold: 1_048_576,
            large_threshold: 104_857_600,
        }
    }
}

impl StrategySelector {
    /// Select optimal strategy for a kernel and input size
    pub fn select<R: Runtime, N: Numeric + CubeElement>(
        &self,
        kernel: &dyn AdaptiveKernel<R, N>,
        shapes: &[Vec<usize>],
    ) -> ExecutionStrategy {
        let total_bytes = kernel.memory_footprint(shapes);
        let total_elements: usize = shapes.iter()
            .map(|s| s.iter().product())
            .max()
            .unwrap_or(0);

        if total_bytes < self.small_threshold {
            // Strategy: Single block + shared memory
            ExecutionStrategy::SingleBlock {
                block_size: 128.min(total_elements as u32),
                shared_mem_bytes: total_bytes,
            }
        } else if total_bytes < self.medium_threshold {
            // Strategy: Multi-block
            let use_shared = kernel.benefits_from_shared(total_elements);
            ExecutionStrategy::MultiBlock {
                block_size: 256,
                num_blocks: ((total_elements + 255) / 256) as u32,
                use_shared,
            }
        } else if total_bytes < self.large_threshold {
            // Strategy: Streaming
            let chunk_size = 1_000_000; // 1M elements per chunk
            ExecutionStrategy::Streaming {
                chunk_size,
                num_chunks: (total_elements + chunk_size - 1) / chunk_size,
            }
        } else {
            // Strategy: Multi-GPU
            ExecutionStrategy::Distributed {
                num_devices: 4, // TODO: Query actual device count
                strategy: DistributionStrategy::DataParallel,
            }
        }
    }

    /// Select strategy specifically for reduction operations
    /// Takes into account parallel vs perpendicular access pattern
    pub fn select_reduction<R: Runtime, N: Numeric + CubeElement>(
        &self,
        kernel: &dyn ReductionKernel<R, N>,
        shape: &[usize],
        axis: usize,
    ) -> (ExecutionStrategy, AccessPattern) {
        let strides = compute_row_major_strides(shape);
        let pattern = kernel.access_pattern(axis, shape, &strides);

        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * std::mem::size_of::<N>();

        let strategy = if total_bytes < self.small_threshold {
            ExecutionStrategy::SingleBlock {
                block_size: 256,
                shared_mem_bytes: total_bytes,
            }
        } else {
            // For reductions, always use shared memory for block-local accumulation
            let num_output_elements: usize = shape.iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .product();

            ExecutionStrategy::MultiBlock {
                block_size: 256,
                num_blocks: num_output_elements as u32,
                use_shared: true,
            }
        };

        (strategy, pattern)
    }
}

// ============================================================================
// SHARED MEMORY PATTERNS
// ============================================================================

/// Trait for kernels that use shared memory
///
/// Shared memory is 10x faster than global memory but limited to ~48-128KB per block
/// See docs/GPU_Programming_Guide.md "Shared Memory - Der Turbo-Boost"
pub trait SharedMemoryKernel<R: Runtime, N: Numeric + CubeElement>: AdaptiveKernel<R, N> {
    /// Calculate shared memory requirement in bytes
    fn shared_memory_size(&self, block_size: u32, line_size: u32) -> usize {
        (block_size * line_size) as usize * std::mem::size_of::<N>()
    }

    /// Use padding to avoid bank conflicts?
    ///
    /// Bank conflicts occur when multiple threads access same memory bank
    /// Add +1 padding to avoid: `SharedMemory::new(block_size + 1)`
    fn use_bank_conflict_padding(&self) -> bool {
        true
    }

    /// Shared memory allocation size with optional padding
    fn padded_shared_size(&self, block_size: u32, line_size: u32) -> usize {
        let base_size = block_size * line_size;
        if self.use_bank_conflict_padding() {
            (base_size + 1) as usize
        } else {
            base_size as usize
        }
    }
}

// ============================================================================
// WARP-LEVEL OPERATIONS
// ============================================================================

/// Trait for kernels using warp-level primitives
///
/// Warps (32 threads on NVIDIA, 64 on AMD) execute in lockstep
/// Warp-level operations avoid explicit synchronization
pub trait WarpKernel<R: Runtime, N: Numeric + CubeElement>: AdaptiveKernel<R, N> {
    /// Warp size (hardware dependent)
    fn warp_size(&self) -> u32 {
        32 // NVIDIA default
    }

    /// Does this kernel use warp shuffle operations?
    fn uses_warp_shuffle(&self) -> bool {
        false
    }

    /// Optimal block size (multiple of warp size)
    fn optimal_block_size(&self) -> u32 {
        let warp = self.warp_size();
        256 // 8 warps on NVIDIA
    }
}

// ============================================================================
// BOUNDS HANDLING PATTERNS
// ============================================================================

/// Strategy for handling out-of-bounds accesses
///
/// See docs/Advanced_CubeCL_Book.md "Bounds Handling Patterns"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsStrategy {
    /// Branch: if/else check (simple but causes warp divergence)
    Branch,

    /// Mask: conditional assignment (no divergence, may do extra work)
    Mask,

    /// Sentinel: pad with special values (e.g., f32::MAX for sorting)
    Sentinel,

    /// None: assume bounds are always valid (unsafe, use with caution!)
    None,
}

pub trait BoundsAware<R: Runtime, N: Numeric + CubeElement>: AdaptiveKernel<R, N> {
    /// How should this kernel handle out-of-bounds accesses?
    fn bounds_strategy(&self) -> BoundsStrategy {
        BoundsStrategy::Branch // Safe default
    }

    /// Sentinel value for padding (if using Sentinel strategy)
    fn sentinel_value(&self) -> N {
        N::from_int(0) // Default: zero
    }

    /// For algorithms requiring power-of-2 sizes (bitonic sort)
    fn requires_power_of_2(&self) -> bool {
        false
    }

    /// Pad size to next power of 2
    fn next_power_of_2(&self, size: usize) -> usize {
        let mut p = 1;
        while p < size {
            p *= 2;
        }
        p
    }
}

// ============================================================================
// KERNEL FUSION SUPPORT
// ============================================================================

/// Trait for element-wise kernels that can be fused
///
/// Fusion combines multiple operations into single kernel launch
/// Benefits: Reduced memory traffic, fewer kernel launches
/// See docs/Advanced_CubeCL_Book.md "Kernel Fusion Pattern"
pub trait FusableKernel<R: Runtime, N: Numeric + CubeElement>: AdaptiveKernel<R, N> {
    /// Can this kernel be fused with another?
    fn can_fuse_with(&self, other_id: &str) -> bool {
        // Default: element-wise ops are generally fusable
        true
    }

    /// Fusion group (ops in same group can be fused together)
    fn fusion_group(&self) -> &'static str {
        "elementwise"
    }
}

pub fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn strategy_to_cube_config(strategy: &ExecutionStrategy) -> (CubeDim, CubeCount) {
    match strategy {
        ExecutionStrategy::SingleBlock { block_size, .. } => {
            (
                CubeDim::new(*block_size, 1, 1),
                CubeCount::Static(1, 1, 1),
            )
        }
        ExecutionStrategy::MultiBlock { block_size, num_blocks, .. } => {
            (
                CubeDim::new(*block_size, 1, 1),
                CubeCount::Static(*num_blocks, 1, 1),
            )
        }
        ExecutionStrategy::Streaming { .. } => {
            
            (
                CubeDim::new(256, 1, 1),
                CubeCount::Static(1, 1, 1),
            )
        }
        ExecutionStrategy::Distributed { .. } => {
            
            (
                CubeDim::new(256, 1, 1),
                CubeCount::Static(1, 1, 1),
            )
        }
    }
}

pub struct KernelConfigBuilder<'a, R: Runtime, N: Numeric + CubeElement> {
    kernel: Option<&'a dyn AdaptiveKernel<R, N>>,
    shapes: Vec<Vec<usize>>,
    reduction_axis: Option<usize>,
    selector: StrategySelector,
}

impl<'a, R: Runtime, N: Numeric + CubeElement> KernelConfigBuilder<'a, R, N> {
    pub fn new() -> Self {
        Self {
            kernel: None,
            shapes: Vec::new(),
            reduction_axis: None,
            selector: StrategySelector::default(),
        }
    }

    pub fn with_kernel(mut self, kernel: &'a dyn AdaptiveKernel<R, N>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    pub fn with_shapes(mut self, shapes: &[Vec<usize>]) -> Self {
        self.shapes = shapes.to_vec();
        self
    }

    pub fn with_reduction_axis(mut self, axis: Option<usize>) -> Self {
        self.reduction_axis = axis;
        self
    }

    pub fn with_selector(mut self, selector: StrategySelector) -> Self {
        self.selector = selector;
        self
    }

    /// Build final configuration
    pub fn build(self) -> KernelLaunchConfig {
        let kernel = self.kernel.expect("Kernel must be set");

        let strategy = self.selector.select(kernel, &self.shapes);
        let (cube_dim, cube_count) = strategy_to_cube_config(&strategy);

        let line_size = if let Some(vk) = self.as_vectorized_kernel(kernel) {
            vk.optimal_line_size()
        } else {
            1
        };

        KernelLaunchConfig {
            cube_dim,
            cube_count,
            line_size,
            strategy,
        }
    }

    fn as_vectorized_kernel(&self, _kernel: &dyn AdaptiveKernel<R, N>) -> Option<&dyn VectorizedKernel<R, N>> {
        // Type casting would go here in full implementation
        None
    }
}

/// Final kernel launch configuration
#[derive(Debug, Clone)]
pub struct KernelLaunchConfig {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub line_size: u32,
    pub strategy: ExecutionStrategy,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        let shape = vec![2, 3, 4];
        let strides = compute_row_major_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_strategy_selector() {
        let selector = StrategySelector::default();

        // Small data -> SingleBlock
        let small_shapes = vec![vec![1024]]; // 4KB for f32
        // Would need actual kernel to test fully

        assert!(selector.small_threshold == 8192);
    }
}
