// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use super::cfg::ComptimeCfg;
use cubecl::prelude::*;

#[cube]
pub fn w0a<N: Numeric + CubePrimitive>(
	mut d: Line<N>,
	#[comptime] steps: u32,
) -> Line<N> {
	#[unroll]
	for s in 0..steps {
		sync_plane();
		let c: Line<N> = plane_shuffle_xor(d, 1u32 << s);
		d = select_many(d.greater_than(c), c, d);
	}
	d
}

#[cube]
pub fn w0d<N: Numeric + CubePrimitive>(
	mut d: Line<N>,
	#[comptime] steps: u32,
) -> Line<N> {
	#[unroll]
	for s in 0..steps {
		sync_plane();
		let c: Line<N> = plane_shuffle_xor(d, 1u32 << s);
		d = select_many(d.less_than(c), c, d);
	}
	d
}

#[cube]
pub fn w0<N: Numeric + CubePrimitive>(
	d: Line<N>,
	#[comptime] steps: u32,
	asc: u32,
) -> Line<N> {
	if asc == 1 {
		w0a::<N>(d, steps)
	} else {
		w0d::<N>(d, steps)
	}
}

#[cube]
pub fn w1a<N: Numeric + CubePrimitive>(
	mut d: Line<N>,
	mut a: Line<u32>,
	#[comptime] steps: u32,
) -> (Line<N>, Line<u32>) {
	#[unroll]
	for s in 0..steps {
		sync_plane();
		let st = 1u32 << s;
		let cd: Line<N> = plane_shuffle_xor(d, st);
		let ca: Line<u32> = plane_shuffle_xor(a, st);
		let sw: Line<bool> = d.greater_than(cd);
		d = select_many(sw, cd, d);
		a = select_many(sw, ca, a);
	}
	(d, a)
}

#[cube]
pub fn w1d<N: Numeric + CubePrimitive>(
	mut d: Line<N>,
	mut a: Line<u32>,
	#[comptime] steps: u32,
) -> (Line<N>, Line<u32>) {
	#[unroll]
	for s in 0..steps {
		sync_plane();
		let st = 1u32 << s;
		let cd: Line<N> = plane_shuffle_xor(d, st);
		let ca: Line<u32> = plane_shuffle_xor(a, st);
		let sw: Line<bool> = d.less_than(cd);
		d = select_many(sw, cd, d);
		a = select_many(sw, ca, a);
	}
	(d, a)
}

#[cube]
pub fn sm0<N: Numeric + CubePrimitive>(
	sm: &mut SharedMemory<Line<N>>,
	#[comptime] ws: u32,
	#[comptime] steps: u32,
) {
	let tid = UNIT_POS;
	let wid = tid / ws;
	let lid = tid % ws;

	#[unroll]
	for stg in 0..steps {
		#[unroll]
		for sub in 0..steps {
			if sub <= stg {
				let ms = stg - sub;
				let wst: u32 = 1u32 << ms;
				let pw = wid ^ wst;
				let pi = pw * ws + lid;

				let bsz = wst * 2u32;
				let bid = wid / bsz;
				let pib = wid % bsz;
				let asc = if (bid % 2u32) == 0u32 {
					pib < wst
				} else {
					pib >= wst
				};

				let my = sm[tid];
				let pr = sm[pi];
				let lo = tid < pi;

				let sw = if asc {
					if lo {
						my.greater_than(pr)
					} else {
						my.less_than(pr)
					}
				} else {
					if lo {
						my.less_than(pr)
					} else {
						my.greater_than(pr)
					}
				};
				sm[tid] = select_many(sw, pr, my);
				sync_cube();
			}
		}
	}
}

#[cube]
pub fn sm1<N: Numeric + CubePrimitive>(
	sd: &mut SharedMemory<Line<N>>,
	sa: &mut SharedMemory<Line<u32>>,
	#[comptime] ws: u32,
	#[comptime] steps: u32,
) {
	let tid = UNIT_POS;
	let wid = tid / ws;
	let lid = tid % ws;

	#[unroll]
	for stg in 0..steps {
		#[unroll]
		for sub in 0..steps {
			if sub <= stg {
				let ms = stg - sub;
				let wst: u32 = 1u32 << ms;
				let pw = wid ^ wst;
				let pi = pw * ws + lid;

				let bsz = wst * 2u32;
				let bid = wid / bsz;
				let pib = wid % bsz;
				let asc = if (bid % 2u32) == 0u32 {
					pib < wst
				} else {
					pib >= wst
				};

				let md = sd[tid];
				let ma = sa[tid];
				let pd = sd[pi];
				let pa = sa[pi];
				let lo = tid < pi;

				let sw = if asc {
					if lo {
						md.greater_than(pd)
					} else {
						md.less_than(pd)
					}
				} else {
					if lo {
						md.less_than(pd)
					} else {
						md.greater_than(pd)
					}
				};
				sd[tid] = select_many(sw, pd, md);
				sa[tid] = select_many(sw, pa, ma);
				sync_cube();
			}
		}
	}
}

#[cube]
pub fn il0c<N: Numeric>(
	sd: &mut SharedMemory<Line<N>>,
	sa: &mut SharedMemory<Line<u32>>,
	tid: u32,
	#[comptime] _ls: u32,
	#[comptime] asc: u32,
) {
	let mut v = sd[tid];
	let mut x = sa[tid];
	let (mut a, mut b, mut c, mut d) = (v[0u32], v[1u32], v[2u32], v[3u32]);
	let (mut ai, mut bi, mut ci, mut di) = (x[0u32], x[1u32], x[2u32], x[3u32]);

	if comptime!(asc == 1) {
		if a > b {
			let t = a;
			a = b;
			b = t;
			let ti = ai;
			ai = bi;
			bi = ti;
		}
		if c > d {
			let t = c;
			c = d;
			d = t;
			let ti = ci;
			ci = di;
			di = ti;
		}
		if a > c {
			let t = a;
			a = c;
			c = t;
			let ti = ai;
			ai = ci;
			ci = ti;
		}
		if b > d {
			let t = b;
			b = d;
			d = t;
			let ti = bi;
			bi = di;
			di = ti;
		}
		if b > c {
			let t = b;
			b = c;
			c = t;
			let ti = bi;
			bi = ci;
			ci = ti;
		}
	} else {
		if a < b {
			let t = a;
			a = b;
			b = t;
			let ti = ai;
			ai = bi;
			bi = ti;
		}
		if c < d {
			let t = c;
			c = d;
			d = t;
			let ti = ci;
			ci = di;
			di = ti;
		}
		if a < c {
			let t = a;
			a = c;
			c = t;
			let ti = ai;
			ai = ci;
			ci = ti;
		}
		if b < d {
			let t = b;
			b = d;
			d = t;
			let ti = bi;
			bi = di;
			di = ti;
		}
		if b < c {
			let t = b;
			b = c;
			c = t;
			let ti = bi;
			bi = ci;
			ci = ti;
		}
	}

	v[0u32] = a;
	v[1u32] = b;
	v[2u32] = c;
	v[3u32] = d;
	x[0u32] = ai;
	x[1u32] = bi;
	x[2u32] = ci;
	x[3u32] = di;
	sd[tid] = v;
	sa[tid] = x;
}

#[cube]
pub fn il0<N: Numeric>(
	sd: &mut SharedMemory<Line<N>>,
	sa: &mut SharedMemory<Line<u32>>,
	tid: u32,
	#[comptime] _ls: u32,
	asc: u32,
) {
	let mut v = sd[tid];
	let mut x = sa[tid];
	let (mut a, mut b, mut c, mut d) = (v[0u32], v[1u32], v[2u32], v[3u32]);
	let (mut ai, mut bi, mut ci, mut di) = (x[0u32], x[1u32], x[2u32], x[3u32]);

	if select(asc == 1, a > b, a < b) {
		let t = a;
		a = b;
		b = t;
		let ti = ai;
		ai = bi;
		bi = ti;
	}
	if select(asc == 1, c > d, c < d) {
		let t = c;
		c = d;
		d = t;
		let ti = ci;
		ci = di;
		di = ti;
	}
	if select(asc == 1, a > c, a < c) {
		let t = a;
		a = c;
		c = t;
		let ti = ai;
		ai = ci;
		ci = ti;
	}
	if select(asc == 1, b > d, b < d) {
		let t = b;
		b = d;
		d = t;
		let ti = bi;
		bi = di;
		di = ti;
	}
	if select(asc == 1, b > c, b < c) {
		let t = b;
		b = c;
		c = t;
		let ti = bi;
		bi = ci;
		ci = ti;
	}

	v[0u32] = a;
	v[1u32] = b;
	v[2u32] = c;
	v[3u32] = d;
	x[0u32] = ai;
	x[1u32] = bi;
	x[2u32] = ci;
	x[3u32] = di;
	sd[tid] = v;
	sa[tid] = x;
}

#[cube]
pub fn il1c<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	#[comptime] _ls: u32,
	#[comptime] asc: u32,
) {
	let mut ln = sm[tid];
	let (mut a, mut b, mut c, mut d) = (ln[0u32], ln[1u32], ln[2u32], ln[3u32]);

	if comptime!(asc == 1) {
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
pub fn il1<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	#[comptime] _ls: u32,
	asc: u32,
) {
	let mut ln = sm[tid];
	let (mut a, mut b, mut c, mut d) = (ln[0u32], ln[1u32], ln[2u32], ln[3u32]);

	if select(asc == 1, a > b, a < b) {
		let t = a;
		a = b;
		b = t;
	}
	if select(asc == 1, c > d, c < d) {
		let t = c;
		c = d;
		d = t;
	}
	if select(asc == 1, a > c, a < c) {
		let t = a;
		a = c;
		c = t;
	}
	if select(asc == 1, b > d, b < d) {
		let t = b;
		b = d;
		d = t;
	}
	if select(asc == 1, b > c, b < c) {
		let t = b;
		b = c;
		c = t;
	}

	ln[0u32] = a;
	ln[1u32] = b;
	ln[2u32] = c;
	ln[3u32] = d;
	sm[tid] = ln;
}

#[cube]
pub fn cl0c<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	#[comptime] sz: u32,
	#[comptime] asc: u32,
) {
	let ns = comptime!(sz.ilog2());
	#[unroll]
	for stg in 0..ns {
		#[unroll]
		for sub in 0..ns {
			if comptime!(sub <= stg) {
				let st = comptime!(1u32 << (stg - sub));
				let p = tid ^ st;
				if p < sz && tid < p {
					let bsz = comptime!(1u32 << (stg + 1));
					let bid = tid / bsz;
					let basc = if comptime!(asc == 1) {
						(bid % 2u32) == 0u32
					} else {
						(bid % 2u32) != 0u32
					};
					let my = sm[tid];
					let pr = sm[p];
					let sw = if basc {
						my.greater_than(pr)
					} else {
						my.less_than(pr)
					};
					sm[tid] = select_many(sw, pr, my);
					sm[p] = select_many(sw, my, pr);
				}
				sync_cube();
			}
		}
	}
}

#[cube]
pub fn cl0<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	#[comptime] sz: u32,
	asc: u32,
) {
	let ns = comptime!(sz.ilog2());
	#[unroll]
	for stg in 0..ns {
		#[unroll]
		for sub in 0..ns {
			if comptime!(sub <= stg) {
				let st = comptime!(1u32 << (stg - sub));
				let p = tid ^ st;
				if p < sz && tid < p {
					let bsz = comptime!(1u32 << (stg + 1));
					let bid = tid / bsz;
					let basc = select(
						asc == 1,
						(bid % 2u32) == 0u32,
						(bid % 2u32) != 0u32,
					);
					let my = sm[tid];
					let pr = sm[p];
					let mk = my[0u32];
					let pk = pr[0u32];
					let sw = select(basc, mk > pk, mk < pk);
					if sw {
						sm[tid] = pr;
						sm[p] = my;
					}
				}
				sync_cube();
			}
		}
	}
}

#[cube]
pub fn cl1<N: Numeric>(
	sm: &mut SharedMemory<Line<N>>,
	tid: u32,
	#[comptime] sz: u32,
	asc: u32,
) {
	cl0::<N>(sm, tid, sz, asc);
}

#[cube]
pub fn wv<N: Numeric + CubePrimitive, D: ListMut<Line<N>>>(
	sm: &D,
	#[comptime] cfg: ComptimeCfg,
	#[comptime] asc: u32,
) {
	let tid = UNIT_POS;
	let wid = tid / cfg.warp_size;
	let ws = comptime!(cfg.warp_size.ilog2());
	let mut d = sm.read(tid);
	let ev = (wid % 2u32) == 0u32;

	if comptime!(asc == 1) {
		if ev {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let c: Line<N> = plane_shuffle_xor(d, 1u32 << s);
				d = select_many(d.greater_than(c), c, d);
			}
		} else {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let c: Line<N> = plane_shuffle_xor(d, 1u32 << s);
				d = select_many(d.less_than(c), c, d);
			}
		}
	} else {
		if ev {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let c: Line<N> = plane_shuffle_xor(d, 1u32 << s);
				d = select_many(d.less_than(c), c, d);
			}
		} else {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let c: Line<N> = plane_shuffle_xor(d, 1u32 << s);
				d = select_many(d.greater_than(c), c, d);
			}
		}
	}
	sm.write(tid, d);
	sync_cube();
}

#[cube]
pub fn wa<N: Numeric + CubePrimitive, D: ListMut<Line<N>>, A: ListMut<Line<u32>>>(
	sd: &D,
	sa: &A,
	#[comptime] cfg: ComptimeCfg,
	asc: u32,
) {
	let tid = UNIT_POS;
	let wid = tid / cfg.warp_size;
	let ws = comptime!(cfg.warp_size.ilog2());
	let mut d = sd.read(tid);
	let mut a = sa.read(tid);
	let ev = (wid % 2u32) == 0u32;

	if 0 == asc {
		if ev {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let st = 1u32 << s;
				let cd: Line<N> = plane_shuffle_xor(d, st);
				let ca: Line<u32> = plane_shuffle_xor(a, st);
				let sw = d.greater_than(cd);
				d = select_many(sw, cd, d);
				a = select_many(sw, ca, a);
			}
		} else {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let st = 1u32 << s;
				let cd: Line<N> = plane_shuffle_xor(d, st);
				let ca: Line<u32> = plane_shuffle_xor(a, st);
				let sw = d.less_than(cd);
				d = select_many(sw, cd, d);
				a = select_many(sw, ca, a);
			}
		}
	} else {
		if ev {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let st = 1u32 << s;
				let cd: Line<N> = plane_shuffle_xor(d, st);
				let ca: Line<u32> = plane_shuffle_xor(a, st);
				let sw = d.less_than(cd);
				d = select_many(sw, cd, d);
				a = select_many(sw, ca, a);
			}
		} else {
			#[unroll]
			for s in 0..ws {
				sync_plane();
				let st = 1u32 << s;
				let cd: Line<N> = plane_shuffle_xor(d, st);
				let ca: Line<u32> = plane_shuffle_xor(a, st);
				let sw = d.greater_than(cd);
				d = select_many(sw, cd, d);
				a = select_many(sw, ca, a);
			}
		}
	}
	sd.write(tid, d);
	sa.write(tid, a);
	sync_cube();
}
