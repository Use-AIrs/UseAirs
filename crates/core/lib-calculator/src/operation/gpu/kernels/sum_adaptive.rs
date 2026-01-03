// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::base::*;
use super::traits::*;

#[derive(Clone, Copy, Debug)]
pub struct SumAdaptive {
    
    pub axis: usize,

    pub shape: Vec<usize>,

    pub force_shared: Option<bool>,
}

impl SumAdaptive {
    
    pub fn new(axis: usize, shape: Vec<usize>) -> Self {
        Self {
            axis,
            shape,
            force_shared: None,
        }
    }

    pub fn with_shared(axis: usize, shape: Vec<usize>) -> Self {
        Self {
            axis,
            shape,
            force_shared: Some(true),
        }
    }
}

impl<R: Runtime, N: Numeric + CubeElement> Kernel<R, N> for SumAdaptive {
    type Cfg = KernelLaunchConfig;
    type Input = GpuMemRep;
    type Output = GpuMemRep;

    fn exec(
        &self,
        order: &KernelOrder<R, N, Self>,
        pool: &GpuMemoryPool<R, N>,
    ) -> Result<()> {
        let config = &order.config;
        let input = &order.input;
        let output = &order.output;

        let input_tuple = pool.get_handles(input)?;
        let output_tuple = pool.get_handles(output)?;

        let tref_in = to_tref(&input_tuple);
        let tref_out = to_tref(&output_tuple);

        let strides = compute_row_major_strides(&self.shape);
        let pattern = self.access_pattern(self.axis, &self.shape, &strides);

        match (&config.strategy, &pattern) {
            
            (ExecutionStrategy::SingleBlock { .. }, AccessPattern::Parallel { .. }) => {
                self.exec_parallel_single_block::<R, N>(
                    pool,
                    tref_in,
                    tref_out,
                    config,
                )
            }

            (ExecutionStrategy::MultiBlock { .. }, AccessPattern::Parallel { .. }) => {
                self.exec_parallel_multi_block::<R, N>(
                    pool,
                    tref_in,
                    tref_out,
                    config,
                )
            }

            (_, AccessPattern::Perpendicular { stride }) => {
                self.exec_perpendicular::<R, N>(
                    pool,
                    tref_in,
                    tref_out,
                    config,
                    *stride,
                )
            }

            _ => {
                reduce::<R, (N, N), N, cubecl_reduce::instructions::Sum>(
                    pool.client(),
                    tref_in,
                    tref_out,
                    self.axis,
                    None,
                    (),
                )
                .map_err(|_| crate::operation::error::OpError::ExecutionError)?;

                Ok(())
            }
        }
    }
}

impl<R: Runtime, N: Numeric + CubeElement> AdaptiveKernel<R, N> for SumAdaptive {
    fn kernel_id(&self) -> &'static str {
        "sum_adaptive"
    }

    fn memory_footprint(&self, _input_shapes: &[Vec<usize>]) -> usize {
        // Input size + output size
        let input_elements: usize = self.shape.iter().product();
        let output_elements: usize = self.shape.iter()
            .enumerate()
            .filter(|(i, _)| *i != self.axis)
            .map(|(_, &s)| s)
            .product();

        (input_elements + output_elements) * std::mem::size_of::<N>()
    }

    fn compute_intensity(&self) -> f32 {
        // Reduction: 1 ADD per element, 1 load, 1/n writes
        // Very memory-bound: ~1 FLOP per 4 bytes
        0.25
    }

    fn benefits_from_shared(&self, total_size: usize) -> bool {
        if let Some(forced) = self.force_shared {
            return forced;
        }

        // Reductions ALWAYS benefit from shared memory for block-local accumulation
        // But only if data size is reasonable (< 8KB per block)
        total_size < 8192
    }

    fn requires_warp_sync(&self) -> bool {
        // Tree reduction requires synchronization
        true
    }
}

// ============================================================================
// VECTORIZED KERNEL TRAIT
// ============================================================================

impl<R: Runtime, N: Numeric + CubeElement> VectorizedKernel<R, N> for SumAdaptive {
    fn optimal_line_size(&self) -> u32 {
        let dtype_size = std::mem::size_of::<N>();
        match dtype_size {
            2 => 8,  // f16: 8 * 2 = 16 bytes
            4 => 4,  // f32: 4 * 4 = 16 bytes
            8 => 2,  // f64: 2 * 8 = 16 bytes
            _ => 1,
        }
    }

    fn supports_vectorization(&self) -> bool {
        // Check if reduction dimension size is multiple of line size
        let line_size = self.optimal_line_size();
        self.shape[self.axis] % line_size as usize == 0
    }
}

// ============================================================================
// REDUCTION KERNEL TRAIT
// ============================================================================

impl<R: Runtime, N: Numeric + CubeElement> ReductionKernel<R, N> for SumAdaptive {
    fn access_pattern(&self, axis: usize, shape: &[usize], strides: &[usize]) -> AccessPattern {
        let rank = shape.len();
        let line_size = self.optimal_line_size();

        // Parallel: last axis in row-major = contiguous memory
        // Example: [M, N] reduce axis=1 -> sum each row
        // Thread access: [a0, a1, a2, a3], [a4, a5, a6, a7] -> COALESCED!
        if axis == rank - 1 {
            AccessPattern::Parallel {
                step_size: line_size * 256, // Assuming 256 threads per block
            }
        } else {
            // Perpendicular: other axes = strided access
            // Example: [M, N] reduce axis=0 -> sum each column
            // Thread access: [a0, b0, c0, d0] with stride=N -> STRIDED but optimized with Lines
            let stride = strides[axis] / line_size as usize;
            AccessPattern::Perpendicular {
                stride: stride as u32,
            }
        }
    }

    fn reduction_stride(&self, axis: usize, strides: &[usize]) -> u32 {
        let line_size = self.optimal_line_size();
        (strides[axis] / line_size as usize) as u32
    }

    fn is_parallel(&self, axis: usize, rank: usize) -> bool {
        axis == rank - 1
    }
}

// ============================================================================
// SHARED MEMORY KERNEL TRAIT
// ============================================================================

impl<R: Runtime, N: Numeric + CubeElement> SharedMemoryKernel<R, N> for SumAdaptive {
    fn shared_memory_size(&self, block_size: u32, line_size: u32) -> usize {
        // Need space for partial sums from each thread
        (block_size * line_size) as usize * std::mem::size_of::<N>()
    }

    fn use_bank_conflict_padding(&self) -> bool {
        // Pad shared memory to avoid bank conflicts
        // Add +1 to array dimension: SharedMemory::new(block_size + 1)
        true
    }

    fn padded_shared_size(&self, block_size: u32, line_size: u32) -> usize {
        let base_size = block_size * line_size;
        // Add padding for bank conflict avoidance
        (base_size + line_size) as usize
    }
}

// ============================================================================
// BOUNDS AWARE TRAIT
// ============================================================================

impl<R: Runtime, N: Numeric + CubeElement> BoundsAware<R, N> for SumAdaptive {
    fn bounds_strategy(&self) -> BoundsStrategy {
        // Use masking to avoid branch divergence
        // See docs/Advanced_CubeCL_Book.md "Bounds Handling Patterns"
        BoundsStrategy::Mask
    }

    fn sentinel_value(&self) -> N {
        // For sum, zero is identity element
        N::from_int(0)
    }

    fn requires_power_of_2(&self) -> bool {
        // Tree reduction works with any size, no power-of-2 requirement
        false
    }
}

// ============================================================================
// EXECUTION IMPLEMENTATIONS
// ============================================================================

impl SumAdaptive {
    /// Execute parallel reduction with single block
    /// Pattern: docs/Advanced_Patterns_Final.md "Small Data (< 8KB)"
    fn exec_parallel_single_block<R: Runtime, N: Numeric + CubeElement>(
        &self,
        pool: &GpuMemoryPool<R, N>,
        input: cubecl_core::prelude::TensorHandleRef<'_, R>,
        output: cubecl_core::prelude::TensorHandleRef<'_, R>,
        config: &KernelLaunchConfig,
    ) -> Result<()> {
        // Use cubecl_reduce with optimized config
        let reduce_config = Config::default()
            .with_use_planes(false)
            .with_use_shared(true);

        reduce::<R, (N, N), N, cubecl_reduce::instructions::Sum>(
            pool.client(),
            input,
            output,
            self.axis,
            Some(reduce_config),
            (),
        )
        .map_err(|_| crate::operation::error::OpError::ExecutionError)?;

        Ok(())
    }

    /// Execute parallel reduction with multiple blocks
    /// Pattern: docs/Advanced_Patterns_Final.md "Medium Data (8KB - 1MB)"
    fn exec_parallel_multi_block<R: Runtime, N: Numeric + CubeElement>(
        &self,
        pool: &GpuMemoryPool<R, N>,
        input: cubecl_core::prelude::TensorHandleRef<'_, R>,
        output: cubecl_core::prelude::TensorHandleRef<'_, R>,
        config: &KernelLaunchConfig,
    ) -> Result<()> {
        // Multi-block reduction with shared memory for block-local reduction
        let reduce_config = Config::default()
            .with_use_planes(false)
            .with_use_shared(true);

        reduce::<R, (N, N), N, cubecl_reduce::instructions::Sum>(
            pool.client(),
            input,
            output,
            self.axis,
            Some(reduce_config),
            (),
        )
        .map_err(|_| crate::operation::error::OpError::ExecutionError)?;

        Ok(())
    }

    /// Execute perpendicular reduction (strided access)
    /// Pattern: docs/Advanced_Patterns_Final.md "Perpendicular Reduction"
    fn exec_perpendicular<R: Runtime, N: Numeric + CubeElement>(
        &self,
        pool: &GpuMemoryPool<R, N>,
        input: cubecl_core::prelude::TensorHandleRef<'_, R>,
        output: cubecl_core::prelude::TensorHandleRef<'_, R>,
        config: &KernelLaunchConfig,
        stride: u32,
    ) -> Result<()> {
        // Perpendicular mode: stride-based access with Line vectorization
        let reduce_config = Config::default()
            .with_use_planes(false)
            .with_use_shared(false); // Shared memory less effective for strided access

        reduce::<R, (N, N), N, cubecl_reduce::instructions::Sum>(
            pool.client(),
            input,
            output,
            self.axis,
            Some(reduce_config),
            (),
        )
        .map_err(|_| crate::operation::error::OpError::ExecutionError)?;

        Ok(())
    }
}

// ============================================================================
// USAGE EXAMPLE
// ============================================================================

#[cfg(test)]
mod example {
    use super::*;

    /// Example: How to use SumAdaptive with automatic strategy selection
    #[test]
    fn example_usage() {
        // 1. Define input shape and reduction axis
        let shape = vec![1024, 1024]; // 1M f32 elements = 4MB
        let axis = 1; // Reduce along columns (parallel mode)

        // 2. Create kernel
        let kernel = SumAdaptive::new(axis, shape.clone());

        // 3. Build configuration with strategy selector
        let selector = StrategySelector::default();

        // 4. Strategy selector automatically chooses:
        // - Line size: 4 (for f32)
        // - Strategy: MultiBlock (4MB > 8KB threshold)
        // - Access pattern: Parallel (axis=1 in row-major)
        // - Block size: 256 threads
        // - Use shared: true (beneficial for reductions)

        // In actual usage:
        // let config = KernelConfigBuilder::new()
        //     .with_kernel(&kernel)
        //     .with_shapes(&[shape])
        //     .with_reduction_axis(Some(axis))
        //     .build();

        // Then launch with:
        // kernel.exec(&order, &pool)?;
    }

    /// Example: Small reduction with single block
    #[test]
    fn example_small_reduction() {
        let shape = vec![256, 8]; // 2KB for f32
        let axis = 1;

        let kernel = SumAdaptive::new(axis, shape);

        // Strategy selector will choose:
        // - Strategy: SingleBlock (2KB < 8KB threshold)
        // - Shared memory: true
        // - All data fits in shared memory
        // - Ultra-fast reduction within single block
    }

    /// Example: Perpendicular reduction
    #[test]
    fn example_perpendicular() {
        let shape = vec![1024, 1024];
        let axis = 0; // Reduce along ROWS (perpendicular to row-major layout)

        let kernel = SumAdaptive::new(axis, shape);

        // Strategy selector detects:
        // - Access pattern: Perpendicular (axis=0 in row-major)
        // - Stride: 1024 / line_size
        // - Uses Line<T> vectorization to optimize strided access
        // - Still efficient despite non-contiguous access!
    }
}

// ============================================================================
// INLINE KERNEL IMPLEMENTATIONS (if needed for custom logic)
// ============================================================================

/// Custom CUDA kernel for ultra-optimized parallel reduction
/// This would be used for very specific optimizations beyond cubecl_reduce
#[cube]
fn sum_kernel_parallel<F: Numeric>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] line_size: u32,
) {
    // Block-level reduction with shared memory
    // See docs/GPU_Programming_Guide.md "Reduction mit Shared Memory"

    let tid = UNIT_POS_X;
    let block_size = CUBE_DIM_X;
    let block_id = CUBE_POS_X;

    // Shared memory for block-local reduction
    let mut shared = SharedMemory::<Line<F>>::new_lined(block_size, line_size);

    // Load data into shared memory
    let idx = block_id * block_size + tid;
    shared[tid] = if idx < input.len() {
        input[idx]
    } else {
        Line::empty(line_size).fill(F::new(0.0))
    };

    sync_units();

    // Tree reduction in shared memory
    let mut stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            shared[tid] += shared[tid + stride];
        }
        sync_units();
        stride /= 2;
    }

    // Thread 0 writes result
    if tid == 0 {
        output[block_id] = shared[0];
    }
}
