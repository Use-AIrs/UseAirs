// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Asc;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Dsc;

pub trait Direction: Clone + Copy + Default + 'static {
	const ASC: u32;
	fn sentinel<N: Numeric>() -> N {
		if Self::ASC == 1 {
			N::max_value()
		} else {
			N::min_value()
		}
	}
}

impl Direction for Asc {
	const ASC: u32 = 1;
}
impl Direction for Dsc {
	const ASC: u32 = 0;
}

#[cube]
pub fn merge_lines<N: Numeric>(
	la: Line<N>,
	lb: Line<N>,
	#[comptime] asc: u32,
) -> (Line<N>, Line<N>) {
	let (mut e0, mut e1, mut e2, mut e3) = (la[0u32], la[1u32], la[2u32], la[3u32]);
	let (mut e4, mut e5, mut e6, mut e7) = (lb[0u32], lb[1u32], lb[2u32], lb[3u32]);

	if comptime!(asc == 1) {
		if e0 > e1 {
			let t = e0;
			e0 = e1;
			e1 = t;
		}
		if e2 > e3 {
			let t = e2;
			e2 = e3;
			e3 = t;
		}
		if e4 > e5 {
			let t = e4;
			e4 = e5;
			e5 = t;
		}
		if e6 > e7 {
			let t = e6;
			e6 = e7;
			e7 = t;
		}
		if e0 > e2 {
			let t = e0;
			e0 = e2;
			e2 = t;
		}
		if e1 > e3 {
			let t = e1;
			e1 = e3;
			e3 = t;
		}
		if e4 > e6 {
			let t = e4;
			e4 = e6;
			e6 = t;
		}
		if e5 > e7 {
			let t = e5;
			e5 = e7;
			e7 = t;
		}
		if e1 > e2 {
			let t = e1;
			e1 = e2;
			e2 = t;
		}
		if e5 > e6 {
			let t = e5;
			e5 = e6;
			e6 = t;
		}
		if e0 > e4 {
			let t = e0;
			e0 = e4;
			e4 = t;
		}
		if e1 > e5 {
			let t = e1;
			e1 = e5;
			e5 = t;
		}
		if e2 > e6 {
			let t = e2;
			e2 = e6;
			e6 = t;
		}
		if e3 > e7 {
			let t = e3;
			e3 = e7;
			e7 = t;
		}
		if e2 > e4 {
			let t = e2;
			e2 = e4;
			e4 = t;
		}
		if e3 > e5 {
			let t = e3;
			e3 = e5;
			e5 = t;
		}
		if e1 > e2 {
			let t = e1;
			e1 = e2;
			e2 = t;
		}
		if e3 > e4 {
			let t = e3;
			e3 = e4;
			e4 = t;
		}
		if e5 > e6 {
			let t = e5;
			e5 = e6;
			e6 = t;
		}
	} else {
		if e0 < e1 {
			let t = e0;
			e0 = e1;
			e1 = t;
		}
		if e2 < e3 {
			let t = e2;
			e2 = e3;
			e3 = t;
		}
		if e4 < e5 {
			let t = e4;
			e4 = e5;
			e5 = t;
		}
		if e6 < e7 {
			let t = e6;
			e6 = e7;
			e7 = t;
		}
		if e0 < e2 {
			let t = e0;
			e0 = e2;
			e2 = t;
		}
		if e1 < e3 {
			let t = e1;
			e1 = e3;
			e3 = t;
		}
		if e4 < e6 {
			let t = e4;
			e4 = e6;
			e6 = t;
		}
		if e5 < e7 {
			let t = e5;
			e5 = e7;
			e7 = t;
		}
		if e1 < e2 {
			let t = e1;
			e1 = e2;
			e2 = t;
		}
		if e5 < e6 {
			let t = e5;
			e5 = e6;
			e6 = t;
		}
		if e0 < e4 {
			let t = e0;
			e0 = e4;
			e4 = t;
		}
		if e1 < e5 {
			let t = e1;
			e1 = e5;
			e5 = t;
		}
		if e2 < e6 {
			let t = e2;
			e2 = e6;
			e6 = t;
		}
		if e3 < e7 {
			let t = e3;
			e3 = e7;
			e7 = t;
		}
		if e2 < e4 {
			let t = e2;
			e2 = e4;
			e4 = t;
		}
		if e3 < e5 {
			let t = e3;
			e3 = e5;
			e5 = t;
		}
		if e1 < e2 {
			let t = e1;
			e1 = e2;
			e2 = t;
		}
		if e3 < e4 {
			let t = e3;
			e3 = e4;
			e4 = t;
		}
		if e5 < e6 {
			let t = e5;
			e5 = e6;
			e6 = t;
		}
	}

	let mut lo = Line::<N>::empty(4u32);
	let mut hi = Line::<N>::empty(4u32);
	lo[0u32] = e0;
	lo[1u32] = e1;
	lo[2u32] = e2;
	lo[3u32] = e3;
	hi[0u32] = e4;
	hi[1u32] = e5;
	hi[2u32] = e6;
	hi[3u32] = e7;
	(lo, hi)
}

#[cube]
pub fn sort_line_inplace<N: Numeric>(
	ln: &mut Line<N>,
	#[comptime] asc: u32,
) {
	let (mut a, mut b, mut c, mut d) = (
		(*ln)[0u32],
		(*ln)[1u32],
		(*ln)[2u32],
		(*ln)[3u32],
	);
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
	(*ln)[0u32] = a;
	(*ln)[1u32] = b;
	(*ln)[2u32] = c;
	(*ln)[3u32] = d;
}
