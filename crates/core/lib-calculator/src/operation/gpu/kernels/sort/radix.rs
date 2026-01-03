// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn rx1_hist(
	input: &Tensor<u32>,
	hist: &mut Tensor<u32>,
	pass: u32,
	n: u32,
	#[comptime] bs: u32,
) {
	let tid = UNIT_POS_X;
	let bid = CUBE_POS_X;
	let gid = bid * bs + tid;

	let mut lh = SharedMemory::<Atomic<u32>>::new(16);

	if tid < 16u32 {
		Atomic::store(&lh[tid], 0u32);
	}
	sync_cube();

	if gid < n {
		let v = input[gid];
		let d = (v >> (pass * 4u32)) & 0xFu32;
		Atomic::add(&lh[d], 1u32);
	}
	sync_cube();

	if tid < 16u32 {
		hist[bid * 16u32 + tid] = Atomic::load(&lh[tid]);
	}
}

#[cube(launch_unchecked)]
pub fn rx2_prefix(
	hist: &Tensor<u32>,
	offs: &mut Tensor<u32>,
	nb: u32,
) {
	let d = UNIT_POS_X;

	let mut tot = SharedMemory::<u32>::new(16);

	if d < 16u32 {
		let mut t = 0u32;
		for b in 0..nb {
			t += hist[b * 16u32 + d];
		}
		tot[d] = t;
	}
	sync_cube();

	if d < 16u32 {
		let mut doff = 0u32;
		for i in 0..d {
			doff += tot[i];
		}

		let mut lsum = 0u32;
		for b in 0..nb {
			let idx = b * 16u32 + d;
			let c = hist[idx];
			offs[idx] = doff + lsum;
			lsum += c;
		}
	}
}

#[cube(launch_unchecked)]
pub fn rx3_scatter(
	input: &Tensor<u32>,
	output: &mut Tensor<u32>,
	offs: &Tensor<u32>,
	pass: u32,
	n: u32,
	#[comptime] bs: u32,
) {
	let tid = UNIT_POS_X;
	let bid = CUBE_POS_X;
	let gid = bid * bs + tid;

	let mut digs = SharedMemory::<u32>::new(bs);

	let my_d = if gid < n {
		let v = input[gid];
		(v >> (pass * 4u32)) & 0xFu32
	} else {
		16u32.into()
	};
	digs[tid] = my_d;
	sync_cube();

	let mut rank = 0u32;
	for i in 0..tid {
		if digs[i] == my_d {
			rank += 1u32;
		}
	}

	if gid < n {
		let v = input[gid];
		let pos = offs[bid * 16u32 + my_d] + rank;
		output[pos] = v;
	}
}

#[cube(launch_unchecked)]
pub fn rx3a_scatter(
	inp_v: &Tensor<u32>,
	inp_i: &Tensor<u32>,
	out_v: &mut Tensor<u32>,
	out_i: &mut Tensor<u32>,
	offs: &Tensor<u32>,
	pass: u32,
	n: u32,
	#[comptime] bs: u32,
) {
	let tid = UNIT_POS_X;
	let bid = CUBE_POS_X;
	let gid = bid * bs + tid;

	let mut digs = SharedMemory::<u32>::new(bs);

	let my_d = if gid < n {
		let v = inp_v[gid];
		(v >> (pass * 4u32)) & 0xFu32
	} else {
		16u32.into()
	};
	digs[tid] = my_d;
	sync_cube();

	let mut rank = 0u32;
	for i in 0..tid {
		if digs[i] == my_d {
			rank += 1u32;
		}
	}

	if gid < n {
		let v = inp_v[gid];
		let idx = inp_i[gid];
		let pos = offs[bid * 16u32 + my_d] + rank;
		out_v[pos] = v;
		out_i[pos] = idx;
	}
}

pub fn f2u(v: f32) -> u32 {
	let b = v.to_bits();
	if b & 0x80000000 != 0 {
		!b
	} else {
		b ^ 0x80000000
	}
}

pub fn u2f(b: u32) -> f32 {
	let b = if b & 0x80000000 != 0 {
		b ^ 0x80000000
	} else {
		!b
	};
	f32::from_bits(b)
}

#[derive(Debug, Clone, Copy)]
pub struct RxCfg {
	pub bs: u32,
	pub nb: u32,
	pub n: u32,
}

impl RxCfg {
	pub fn new(n: u32) -> Self {
		let bs = 256u32;
		let nb = (n + bs - 1) / bs;
		Self { bs, nb, n }
	}

	pub fn hist_sz(&self) -> usize {
		(self.nb * 16) as usize
	}
}
