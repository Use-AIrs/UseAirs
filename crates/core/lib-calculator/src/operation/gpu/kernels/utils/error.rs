// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use core::fmt;

use cubecl_core::ir::StorageType;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ReduceError {
	PlanesUnavailable,

	CubeCountTooLarge,

	ImprecisePlaneDim,

	InvalidAxis {
		axis: usize,
		rank: usize,
	},

	MismatchShape {
		expected_shape: Vec<usize>,
		output_shape: Vec<usize>,
	},

	MissingAtomicAdd(StorageType),
}

impl fmt::Display for ReduceError {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		match self {
            Self::PlanesUnavailable => write!(
                f,
                "Trying to launch a kernel using plane instructions, but there are not supported by the hardware."
            ),
            Self::CubeCountTooLarge => {
                write!(f, "The cube count is larger than the max supported.")
            }
            Self::ImprecisePlaneDim => write!(
                f,
                "Trying to launch a kernel using plane instructions, but the min and max plane dimensions are different."
            ),
            Self::InvalidAxis { axis, rank } => write!(
                f,
                "The provided axis ({axis}) must be smaller than the input tensor rank ({rank})."
            ),
            Self::MismatchShape {
                expected_shape,
                output_shape,
            } => {
                write!(
                    f,
                    "The output shape (currently {output_shape:?}) should be {expected_shape:?}."
                )
            }
            Self::MissingAtomicAdd(elem) => {
                write!(f, "Atomic add not supported by the client for {elem}")
            }
        }
	}
}
