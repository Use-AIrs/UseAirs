// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use crate::operation::gpu::kernels::sort::cfg::ComptimeCfg;
use crate::operation::gpu::kernels::sort::range::{k0_ranger, K0Range, SortPhase};
use cubecl::prelude::*;

#[cube]
pub fn sort_values<N: Numeric>(
	inp: &Slice<Line<N>>,
	out: &mut Slice<Line<N>, ReadWrite>,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) {
	let rng = k0_ranger(SortPhase::IntroSort, cfg, asc);
	let sl = comptime!(cfg.shmem_len);
	let ls = comptime!(cfg.line_size);
	let mut sm = SharedMemory::<N>::new_lined(sl, ls);

	ld::<N>(inp, &mut sm, &rng, cfg);
	sync_cube();
	ssb::<N>(&mut sm, &rng, cfg);
	sync_cube();
	wr::<N>(&sm, out, &rng, cfg);
}

#[cube]
fn ld<N: Numeric>(
	inp: &Slice<Line<N>>,
	sm: &mut SharedMemory<Line<N>>,
	rng: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let sl = comptime!(cfg.shmem_len);
	let tl = comptime!(cfg.row_size);
	let tid = UNIT_POS_X;
	let gi = rng.chunk_offset + tid;
	let snt = select(
		rng.ascending(),
		N::max_value(),
		N::min_value(),
	);
	let sln = Line::<N>::new(snt);
	if tid < sl {
		let v = gi < tl;
		let si = gi * u32::cast_from(v);
		sm[tid] = select(v, inp[si], sln);
	}
}

#[cube]
fn wr<N: Numeric>(
	sm: &SharedMemory<Line<N>>,
	out: &mut Slice<Line<N>, ReadWrite>,
	rng: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let sl = comptime!(cfg.shmem_len);
	let tl = comptime!(cfg.row_size);
	let tid = UNIT_POS_X;
	let gi = rng.chunk_offset + tid;
	if tid < sl && gi < tl {
		out[gi] = sm[tid];
	}
}

#[cube]
fn ssb<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	rng: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let sl = comptime!(cfg.shmem_len);
	let ls = comptime!(cfg.line_size);
	let ns = comptime!(sl.ilog2());
	let tid = UNIT_POS_X;

	if tid < sl {
		ils::<N>(sm, tid, rng.is_ascending, ls);
	}
	sync_cube();

	#[unroll]
	for stg in 0..ns {
		#[unroll]
		for sub in 0..ns {
			if comptime!(sub <= stg) {
				let ms = comptime!(stg - sub);
				bms::<N>(sm, tid, stg, ms, rng.is_ascending, cfg);
				sync_cube();
			}
		}
	}

	if tid < sl {
		ils::<N>(sm, tid, rng.is_ascending, ls);
	}
}

#[cube]
fn bms<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	#[comptime] stg: u32,
	#[comptime] ms: u32,
	ca: u32,
	#[comptime] cfg: ComptimeCfg,
) {
	let sl = comptime!(cfg.shmem_len);
	let str = comptime!(1u32 << ms);
	let bs = comptime!(1u32 << (stg + 1));
	let p = tid ^ str;

	if p < sl && tid < p {
		let bi = tid / bs;
		let ba = select(
			ca == 1,
			(bi % 2u32) == 0u32,
			(bi % 2u32) != 0u32,
		);
		let ml = sm[tid];
		let pl = sm[p];
		let mk = ml[0u32];
		let pk = pl[0u32];
		let sw = select(ba, mk > pk, mk < pk);
		if sw {
			sm[tid] = pl;
			sm[p] = ml;
		}
	}
}

#[cube]
fn ils<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	asc: u32,
	#[comptime] ls: u32,
) {
	let mut ln = sm[tid];
	let (mut a, mut b, mut c, mut d) = (ln[0u32], ln[1u32], ln[2u32], ln[3u32]);

	if asc == 1 {
		if a > b {
			let t = a;
			a = b;
			b = t;
		}
		if c > d {
			let t = c;
			c = d;
			d = t;
		}
		if a > c {
			let t = a;
			a = c;
			c = t;
		}
		if b > d {
			let t = b;
			b = d;
			d = t;
		}
		if b > c {
			let t = b;
			b = c;
			c = t;
		}
	} else {
		if a < b {
			let t = a;
			a = b;
			b = t;
		}
		if c < d {
			let t = c;
			c = d;
			d = t;
		}
		if a < c {
			let t = a;
			a = c;
			c = t;
		}
		if b < d {
			let t = b;
			b = d;
			d = t;
		}
		if b < c {
			let t = b;
			b = c;
			c = t;
		}
	}

	ln[0u32] = a;
	ln[1u32] = b;
	ln[2u32] = c;
	ln[3u32] = d;
	sm[tid] = ln;
}

#[cube]
pub fn to_out_sp<N: Numeric>(
	sm: &SharedMemory<Line<N>>,
	out: &mut Slice<Line<N>, ReadWrite>,
	rng: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	let sl = comptime!(cfg.shmem_len);
	let tl = comptime!(cfg.row_size);
	let tid = UNIT_POS_X;
	let gi = rng.chunk_offset + tid;

	if tid < sl && gi < tl {
		let ln = sm[tid];
		let (mut a, mut b, mut c, mut d) = (ln[0u32], ln[1u32], ln[2u32], ln[3u32]);

		if rng.ascending() {
			if a > b {
				let t = a;
				a = b;
				b = t;
			}
			if c > d {
				let t = c;
				c = d;
				d = t;
			}
			if a > c {
				let t = a;
				a = c;
				c = t;
			}
			if b > d {
				let t = b;
				b = d;
				d = t;
			}
			if b > c {
				let t = b;
				b = c;
				c = t;
			}
		} else {
			if a < b {
				let t = a;
				a = b;
				b = t;
			}
			if c < d {
				let t = c;
				c = d;
				d = t;
			}
			if a < c {
				let t = a;
				a = c;
				c = t;
			}
			if b < d {
				let t = b;
				b = d;
				d = t;
			}
			if b < c {
				let t = b;
				b = c;
				c = t;
			}
		}

		let mut ol = Line::<N>::empty(4u32);
		ol[0u32] = a;
		ol[1u32] = b;
		ol[2u32] = c;
		ol[3u32] = d;
		out[gi] = ol;
	}
}

#[cube]
pub fn to_out_pp<N: Numeric>(
	sm: &SharedMemory<Line<N>>,
	out: &mut Slice<Line<N>, ReadWrite>,
	rng: &K0Range,
	#[comptime] cfg: ComptimeCfg,
) {
	to_out_sp::<N>(sm, out, rng, cfg);
}

#[cube]
pub fn to_out_fin<N: Numeric>(
	inp: &Slice<Line<N>>,
	out: &mut Slice<Line<N>, ReadWrite>,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) {
	let tl = comptime!(cfg.row_size);
	let tid = UNIT_POS_X;
	let gi = CUBE_POS_X * comptime!(cfg.shmem_len) + tid;

	if gi < tl {
		let ln = inp[gi];
		let (mut a, mut b, mut c, mut d) = (ln[0u32], ln[1u32], ln[2u32], ln[3u32]);

		if asc == 1 {
			if a > b {
				let t = a;
				a = b;
				b = t;
			}
			if c > d {
				let t = c;
				c = d;
				d = t;
			}
			if a > c {
				let t = a;
				a = c;
				c = t;
			}
			if b > d {
				let t = b;
				b = d;
				d = t;
			}
			if b > c {
				let t = b;
				b = c;
				c = t;
			}
		} else {
			if a < b {
				let t = a;
				a = b;
				b = t;
			}
			if c < d {
				let t = c;
				c = d;
				d = t;
			}
			if a < c {
				let t = a;
				a = c;
				c = t;
			}
			if b < d {
				let t = b;
				b = d;
				d = t;
			}
			if b < c {
				let t = b;
				b = c;
				c = t;
			}
		}

		let mut ol = Line::<N>::empty(4u32);
		ol[0u32] = a;
		ol[1u32] = b;
		ol[2u32] = c;
		ol[3u32] = d;
		out[gi] = ol;
	}
}
