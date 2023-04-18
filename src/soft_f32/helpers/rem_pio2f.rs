/* origin: FreeBSD /usr/src/lib/msun/src/e_rem_pio2f.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Debugged and optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

use crate::{
    soft_f32::SoftF32,
    soft_f64::{helpers::rem_pio2_large, SoftF64},
};

const TOINT: SoftF64 = SoftF64(1.5).div(SoftF64(f64::EPSILON));

/// 53 bits of 2/pi
const INV_PIO2: SoftF64 = SoftF64(6.36619772367581382433e-01); /* 0x3FE45F30, 0x6DC9C883 */
/// first 25 bits of pi/2
const PIO2_1: SoftF64 = SoftF64(1.57079631090164184570e+00); /* 0x3FF921FB, 0x50000000 */
/// pi/2 - pio2_1
const PIO2_1T: SoftF64 = SoftF64(1.58932547735281966916e-08); /* 0x3E5110b4, 0x611A6263 */

/// Return the remainder of x rem pi/2 in *y
///
/// use double precision for everything except passing x
/// use __rem_pio2_large() for large x
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub(crate) const fn rem_pio2f(x: SoftF32) -> (i32, SoftF64) {
    let x64 = SoftF64(x.0 as f64);

    let mut tx: [SoftF64; 1] = [SoftF64(0.0)];
    let ty: [SoftF64; 1] = [SoftF64(0.0)];

    let ix = x.to_bits() & 0x7fffffff;
    /* 25+53 bit pi is good enough for medium size */
    if ix < 0x4dc90fdb {
        /* |x| ~< 2^28*(pi/2), medium size */
        /* Use a specialized rint() to get fn.  Assume round-to-nearest. */
        let tmp = x64.mul(INV_PIO2).add(TOINT);

        let f_n = tmp.sub(TOINT);
        return (f_n.0 as i32, x64.sub(f_n.mul(PIO2_1)).sub(f_n.mul(PIO2_1T)));
    }
    if ix >= 0x7f800000 {
        /* x is inf or NaN */
        return (0, x64.sub(x64));
    }
    /* scale x into [2^23, 2^24-1] */
    let sign = (x.to_bits() >> 31) != 0;
    let e0 = ((ix >> 23) - (0x7f + 23)) as i32; /* e0 = ilogb(|x|)-23, positive */
    tx[0] = SoftF64(SoftF32::from_bits(ix - (e0 << 23) as u32).0 as f64);
    let (n, ty) = rem_pio2_large(&tx, &ty, e0, 0);
    if sign {
        return (-n, ty[0].neg());
    }
    (n, ty[0])
}
