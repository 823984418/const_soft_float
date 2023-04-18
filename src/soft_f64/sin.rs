// origin: FreeBSD /usr/src/lib/msun/src/s_sin.c, https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/sin.rs */
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

use super::{
    helpers::{k_cos, k_sin, rem_pio2},
    SoftF64,
};

// sin(x)
// Return sine function of x.
//
// kernel function:
//      k_sin            ... sine function on [-pi/4,pi/4]
//      k_cos            ... cose function on [-pi/4,pi/4]
//      rem_pio2         ... argument reduction routine
//
// Method.rounded
//      Let S,C and T denote the sin, cos and tan respectively on
//      [-PI/4, +PI/4]. Reduce the argument x to y1+y2 = x-k*pi/2
//      in [-pi/4 , +pi/4], and let n = k mod 4.
//      We have
//
//          n        sin(x)      cos(x)        tan(x)
//     ----------------------------------------------------------
//          0          S           C             T
//          1          C          -S            -1/T
//          2         -S          -C             T
//          3         -C           S            -1/T
//     ----------------------------------------------------------
//
// Special cases:
//      Let trig be any of sin, cos, or tan.
//      trig(+-INF)  is NaN, with signals;
//      trig(NaN)    is that NaN;
//
// Accuracy:
//      TRIG(x) returns trig(x) nearly rounded
pub(crate) const fn sin(x: SoftF64) -> SoftF64 {
    let x1p120 = SoftF64::from_bits(0x4770000000000000); // 0x1p120f === 2 ^ 120

    /* High word of x. */
    let ix = (SoftF64::to_bits(x) >> 32) as u32 & 0x7fffffff;

    /* |x| ~< pi/4 */
    if ix <= 0x3fe921fb {
        if ix < 0x3e500000 {
            /* |x| < 2**-26 */
            /* raise inexact if x != 0 and underflow if subnormal*/
            if ix < 0x00100000 {
                x.div(x1p120);
            } else {
                x.add(x1p120);
            }
            return x;
        }
        return k_sin(x, SoftF64::ZERO, 0);
    }

    /* sin(Inf or NaN) is NaN */
    if ix >= 0x7ff00000 {
        return x.sub(x);
    }

    /* argument reduction needed */
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_sin(y0, y1, 1),
        1 => k_cos(y0, y1),
        2 => k_sin(y0, y1, 1).neg(),
        _ => k_cos(y0, y1).neg(),
    }
}

#[cfg(test)]
mod test {
    use crate::soft_f64::SoftF64;

    #[test]
    fn test_near_pi() {
        let x = SoftF64::from_bits(0x400921fb000FD5DD); // 3.141592026217707
        let sx = SoftF64::from_bits(0x3ea50d15ced1a4a2); // 6.273720864039205e-7
        let result = x.sin().0;
        assert_eq!(result, sx.0);
    }

    #[test]
    fn test_large_neg() {
        assert_eq!(SoftF64(-1647101.0).sin().to_f64(), (-1647101.0_f64).sin())
    }
}
