// SPDX-License-Identifier: LicenseRef-PolyForm-Perimeter-1.0.1
// Copyright (c) 2026 Use-AI.rs
//
// This file is part of Use-Ai.rs
// See LICENSE for details

use cubecl::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use crate::operation::gpu::kernels::sort::cfg::ComptimeCfg;
use crate::operation::gpu::kernels::sort::alg::{il0, cl0};

#[cube]
pub fn sort_values<N: Numeric, I: SliceOperator<Line<N>>, O: SliceMutOperator<Line<N>>>(
    inp: &I, out: &mut O, #[comptime] cfg: ComptimeCfg, asc: u32,
) {
    let mc = comptime!(cfg.cycles_per_axis > 1);
    if comptime!(mc) {
        sv_mc::<N>(&inp.to_slice(), &mut out.to_slice_mut(), cfg, asc);
    } else {
        sv_sp::<N>(&inp.to_slice(), &mut out.to_slice_mut(), cfg, asc);
    }
}

#[cube]
fn sv_sp<N: Numeric>(inp: &Slice<Line<N>>, out: &mut Slice<Line<N>, ReadWrite>, #[comptime] cfg: ComptimeCfg, asc: u32) {
    let sl = comptime!(cfg.shmem_len);
    let rs = comptime!(cfg.row_size);
    let ls = comptime!(cfg.line_size);
    let tid = UNIT_POS_X;
    let ro = CUBE_POS_X * rs;

    let mut sm = SharedMemory::<Line<N>>::new(sl);
    let snt = select(asc == 1, N::max_value(), N::min_value());
    let sln = Line::<N>::new(snt);
    let gi = ro + tid;

    if tid < sl { sm[tid] = sln; }
    if tid < rs { sm[tid] = inp[gi]; }
    sync_cube();

    il0::<N>(&mut sm, tid, ls, asc);
    sync_cube();

    let ns = comptime!(sl.ilog2());
    #[unroll]
    for stg in 0..ns {
        #[unroll]
        for sub in 0..ns {
            if comptime!(sub <= stg) {
                let ms = comptime!(stg - sub);
                let str = comptime!(1u32 << ms);
                let p = tid ^ str;
                if p < sl && tid < p {
                    let bs = comptime!(1u32 << (stg + 1));
                    let bi = tid / bs;
                    let ba = select(asc==1, (bi%2u32)==0u32, (bi%2u32)!=0u32);
                    let ml = sm[tid];
                    let pl = sm[p];
                    let mk = ml[0u32];
                    let pk = pl[0u32];
                    let sw = select(ba, mk>pk, mk<pk);
                    if sw { sm[tid]=pl; sm[p]=ml; }
                }
                sync_cube();
            }
        }
    }

    if tid < rs { out[gi] = sm[tid]; }
}

#[cube]
fn sv_mc<N: Numeric>(inp: &Slice<Line<N>>, out: &mut Slice<Line<N>, ReadWrite>, #[comptime] cfg: ComptimeCfg, asc: u32) {
    let cid = CUBE_POS_X;
    let ca = select((cid % 2u32) == 0u32, asc, 1u32 - asc);
    sv_mck::<N>(inp, out, cfg, ca);
}

#[cube]
pub fn sv_mck<N: Numeric>(inp: &Slice<Line<N>>, out: &mut Slice<Line<N>, ReadWrite>, #[comptime] cfg: ComptimeCfg, asc: u32) {
    let sl = comptime!(cfg.shmem_len);
    let ls = comptime!(cfg.line_size);
    let tl = comptime!(cfg.row_size);
    let tid = UNIT_POS_X;
    let cid = CUBE_POS_X;
    let co = cid * sl;
    let gi = co + tid;

    let mut sm = SharedMemory::<Line<N>>::new(sl);
    let snt = select(asc == 1, N::max_value(), N::min_value());
    let sln = Line::<N>::new(snt);

    if tid < sl { sm[tid] = sln; }
    if tid < sl && gi < tl { sm[tid] = inp[gi]; }
    sync_cube();

    il0::<N>(&mut sm, tid, ls, asc);
    sync_cube();

    cl0::<N>(&mut sm, tid, sl, asc);

    il0::<N>(&mut sm, tid, ls, asc);
    sync_cube();

    if tid < sl && gi < tl { out[gi] = sm[tid]; }
}
