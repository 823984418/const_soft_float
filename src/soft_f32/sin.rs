/* origin: FreeBSD /usr/src/lib/msun/src/s_sinf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Optimized by Bruce D. Evans.
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

use core::f64::consts::FRAC_PI_2;

use crate::soft_f64::SoftF64;

use super::{
    helpers::{k_cosf, k_sinf, rem_pio2f},
    SoftF32,
};

/* Small multiples of pi/2 rounded to double precision. */
const S1_PIO2: SoftF64 = SoftF64(1.).mul(SoftF64(FRAC_PI_2)); /* 0x3FF921FB, 0x54442D18 */
const S2_PIO2: SoftF64 = SoftF64(2.).mul(SoftF64(FRAC_PI_2)); /* 0x400921FB, 0x54442D18 */
const S3_PIO2: SoftF64 = SoftF64(3.).mul(SoftF64(FRAC_PI_2)); /* 0x4012D97C, 0x7F3321D2 */
const S4_PIO2: SoftF64 = SoftF64(4.).mul(SoftF64(FRAC_PI_2)); /* 0x401921FB, 0x54442D18 */

pub const fn sinf(x: SoftF32) -> SoftF32 {
    let x64 = SoftF64(x.0 as f64);

    let x1p120 = SoftF32::from_bits(0x7b800000); // 0x1p120f === 2 ^ 120

    let mut ix = x.to_bits();
    let sign = (ix >> 31) != 0;
    ix &= 0x7fffffff;

    if ix <= 0x3f490fda {
        /* |x| ~<= pi/4 */
        if ix < 0x39800000 {
            /* |x| < 2**-12 */
            /* raise inexact if x!=0 and underflow if subnormal */
            if ix < 0x00800000 {
                let _ = x.div(x1p120);
            } else {
                let _ = x.add(x1p120);
            };
            return x;
        }
        return k_sinf(x64);
    }
    if ix <= 0x407b53d1 {
        /* |x| ~<= 5*pi/4 */
        if ix <= 0x4016cbe3 {
            /* |x| ~<= 3pi/4 */
            if sign {
                return k_cosf(x64.add(S1_PIO2)).neg();
            } else {
                return k_cosf(x64.sub(S1_PIO2));
            }
        }
        return k_sinf(if sign {
            x64.add(S2_PIO2).neg()
        } else {
            x64.sub(S2_PIO2).neg()
        });
    }
    if ix <= 0x40e231d5 {
        /* |x| ~<= 9*pi/4 */
        if ix <= 0x40afeddf {
            /* |x| ~<= 7*pi/4 */
            if sign {
                return k_cosf(x64.add(S3_PIO2));
            } else {
                return k_cosf(x64.sub(S3_PIO2)).neg();
            }
        }
        return k_sinf(if sign {
            x64.add(S4_PIO2)
        } else {
            x64.sub(S4_PIO2)
        });
    }

    /* sin(Inf or NaN) is NaN */
    if ix >= 0x7f800000 {
        return x.sub(x);
    }

    /* general argument reduction needed */
    let (n, y) = rem_pio2f(x);
    match n & 3 {
        0 => k_sinf(y),
        1 => k_cosf(y),
        2 => k_sinf(y.neg()),
        _ => k_cosf(y).neg(),
    }
}

#[cfg(test)]
mod test {
    use core::f32::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_3, PI};

    use super::*;

    #[test]
    fn test_basic() {
        for val in [0.0, FRAC_PI_3, FRAC_PI_2, PI, FRAC_2_PI] {
            assert_eq!(SoftF32(val).sin().to_f32(), val.sin())
        }
    }
}
