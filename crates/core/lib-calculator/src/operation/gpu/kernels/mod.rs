// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

mod argmax;
mod argmin;
mod base;

#[cfg(feature = "nccl")]
mod nccl;
mod negate;
mod one_minus;
mod padding;
mod prod;

pub mod sort;
mod sum;
mod to_tensor;
pub(crate) mod utils;
mod zip;

pub use argmax::ArgMax;
pub use argmin::ArgMin;

#[cfg(feature = "nccl")]
pub use nccl::{AllGather, AllReduce, Broadcast, NcclInitialize, ReduceScatter};
pub use negate::Negate;
pub use one_minus::OneMinus;
pub use padding::Padding;
pub use prod::Prod;
pub use sort::{
	bitonic_argsort_kernel, bitonic_sort_kernel, f2u, rx1_hist, rx2_prefix, rx3_scatter,
	rx3a_scatter, u2f, ArgSort, ComptimeCfg, RxCfg, SCfg, Sort, SortConfig, SortStrategy,
};
pub use sum::Sum;
pub use to_tensor::ToTensor;
pub use zip::Zip;

pub use base::{Kernel, KernelOrder};
