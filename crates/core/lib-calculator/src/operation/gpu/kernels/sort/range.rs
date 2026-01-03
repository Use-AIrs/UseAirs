// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::cfg::ComptimeCfg;
use cubecl::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SortPhase {
	IntroSort,
	GlobalMerge { stage: u32, substep: u32 },
}

#[derive(CubeType, Clone, Copy)]
pub struct K0Range {
	pub chunk_id: u32,
	pub chunk_offset: u32,
	pub partner_chunk_id: u32,
	pub partner_offset: u32,
	pub is_ascending: u32,
	pub i_am_lower: u32,
}

#[cube]
pub fn k0_ranger(
	#[comptime] phase: SortPhase,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) -> K0Range {
	match comptime!(phase) {
		SortPhase::IntroSort => k0r_intro(cfg, asc),
		SortPhase::GlobalMerge { stage, substep } => k0r_gm(stage, substep, cfg, asc),
	}
}

#[cube]
fn k0r_intro(
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) -> K0Range {
	let sl = comptime!(cfg.shmem_len);
	let cid = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
	let co = cid * sl;
	let ia = if asc == 1 {
		select((cid % 2u32) == 0u32, 1u32, 0u32)
	} else {
		select((cid % 2u32) != 0u32, 1u32, 0u32)
	};
	K0Range {
		chunk_id: cid,
		chunk_offset: co,
		partner_chunk_id: 0,
		partner_offset: 0,
		is_ascending: ia,
		i_am_lower: 1,
	}
}

#[cube]
fn k0r_gm(
	#[comptime] stg: u32,
	#[comptime] sub: u32,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) -> K0Range {
	let sl = comptime!(cfg.shmem_len);
	let cid = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
	let co = cid * sl;
	let str = comptime!(1u32 << (stg - sub));
	let pc = cid ^ str;
	let po = pc * sl;
	let bbs = comptime!(1u32 << (stg + 1));
	let bid = cid / bbs;
	let ia = if asc == 1 {
		select((bid % 2u32) == 0u32, 1u32, 0u32)
	} else {
		select((bid % 2u32) != 0u32, 1u32, 0u32)
	};
	let lo = select(cid < pc, 1u32, 0u32);
	K0Range {
		chunk_id: cid,
		chunk_offset: co,
		partner_chunk_id: pc,
		partner_offset: po,
		is_ascending: ia,
		i_am_lower: lo,
	}
}

#[cube]
impl K0Range {
	pub fn partner_valid(
		&self,
		#[comptime] nc: u32,
	) -> bool {
		self.partner_chunk_id < nc
	}
	pub fn ascending(&self) -> bool {
		self.is_ascending == 1
	}
	pub fn is_lower(&self) -> bool {
		self.i_am_lower == 1
	}

	pub fn my_slice<N: CubePrimitive, I: SliceOperator<Line<N>>>(
		&self,
		inp: &I,
		#[comptime] cs: u32,
	) -> Slice<Line<N>> {
		inp.slice(self.chunk_offset, self.chunk_offset + cs)
	}

	pub fn partner_slice<N: CubePrimitive, I: SliceOperator<Line<N>>>(
		&self,
		inp: &I,
		#[comptime] cs: u32,
	) -> Slice<Line<N>> {
		inp.slice(
			self.partner_offset,
			self.partner_offset + cs,
		)
	}

	pub fn my_slice_mut<N: CubePrimitive, O: SliceMutOperator<Line<N>>>(
		&self,
		out: &mut O,
		#[comptime] cs: u32,
	) -> Slice<Line<N>, ReadWrite> {
		out.slice_mut(self.chunk_offset, self.chunk_offset + cs)
	}
}
