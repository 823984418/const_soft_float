/* origin: FreeBSD /usr/src/lib/msun/src/e_sqrt.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* sqrt(x)
 * Return correctly rounded sqrt.
 *           ------------------------------------------
 *           |  Use the hardware sqrt if you have one |
 *           ------------------------------------------
 * Method:
 *   Bit by bit method using integer arithmetic. (Slow, but portable)
 *   1. Normalization
 *      Scale x to y in [1,4) with even powers of 2:
 *      find an integer k such that  1 <= (y=x*2^(2k)) < 4, then
 *              sqrt(x) = 2^k * sqrt(y)
 *   2. Bit by bit computation
 *      Let q  = sqrt(y) truncated to i bit after binary point (q = 1),
 *           i                                                   0
 *                                     i+1         2
 *          s  = 2*q , and      y  =  2   * ( y - q  ).         (1)
 *           i      i            i                 i
 *
 *      To compute q    from q , one checks whether
 *                  i+1       i
 *
 *                            -(i+1) 2
 *                      (q + 2      ) <= y.                     (2)
 *                        i
 *                                                            -(i+1)
 *      If (2) is false, then q   = q ; otherwise q   = q  + 2      .
 *                             i+1   i             i+1   i
 *
 *      With some algebraic manipulation, it is not difficult to see
 *      that (2) is equivalent to
 *                             -(i+1)
 *                      s  +  2       <= y                      (3)
 *                       i                i
 *
 *      The advantage of (3) is that s  and y  can be computed by
 *                                    i      i
 *      the following recurrence formula:
 *          if (3) is false
 *
 *          s     =  s  ,       y    = y   ;                    (4)
 *           i+1      i          i+1    i
 *
 *          otherwise,
 *                         -i                     -(i+1)
 *          s     =  s  + 2  ,  y    = y  -  s  - 2             (5)
 *           i+1      i          i+1    i     i
 *
 *      One may easily use induction to prove (4) and (5).
 *      Note. Since the left hand side of (3) contain only i+2 bits,
 *            it does not necessary to do a full (53-bit) comparison
 *            in (3).
 *   3. Final rounding
 *      After generating the 53 bits result, we compute one more bit.
 *      Together with the remainder, we can decide whether the
 *      result is exact, bigger than 1/2ulp, or less than 1/2ulp
 *      (it will never equal to 1/2ulp).
 *      The rounding mode can be detected by checking whether
 *      huge + tiny is equal to huge, and whether huge - tiny is
 *      equal to huge for some floating point number "huge" and "tiny".
 *
 * Special cases:
 *      sqrt(+-0) = +-0         ... exact
 *      sqrt(inf) = inf
 *      sqrt(-ve) = NaN         ... with invalid signal
 *      sqrt(NaN) = NaN         ... with invalid signal for signaling NaN
 */

use crate::soft_f64::{
    helpers::{ge, gt},
    SoftF64,
};

type F = SoftF64;

pub(crate) const fn sqrt(x: F) -> F {
    const TINY: F = SoftF64(1.0e-300);

    let mut z: F;
    let sign: u32 = 0x80000000;
    let mut ix0: i32;
    let mut s0: i32;
    let mut q: i32;
    let mut m: i32;
    let mut t: i32;
    let mut i: i32;
    let mut r: u32;
    let mut t1: u32;
    let mut s1: u32;
    let mut ix1: u32;
    let mut q1: u32;

    ix0 = (x.to_bits() >> 32) as i32;
    ix1 = x.to_bits() as u32;

    /* take care of Inf and NaN */
    if (ix0 & 0x7ff00000) == 0x7ff00000 {
        return x.mul(x).add(x); /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
    }
    /* take care of zero */
    if ix0 <= 0 {
        if ((ix0 & !(sign as i32)) | ix1 as i32) == 0 {
            return x; /* sqrt(+-0) = +-0 */
        }
        if ix0 < 0 {
            return (x.sub(x)).div(x.sub(x)); /* sqrt(-ve) = sNaN */
        }
    }
    /* normalize x */
    m = ix0 >> 20;
    if m == 0 {
        /* subnormal x */
        while ix0 == 0 {
            m -= 21;
            ix0 |= (ix1 >> 11) as i32;
            ix1 <<= 21;
        }
        i = 0;
        while (ix0 & 0x00100000) == 0 {
            i += 1;
            ix0 <<= 1;
        }
        m -= i - 1;
        ix0 |= (ix1 as usize >> (32 - i) as usize) as i32;
        ix1 = ix1 << i as usize;
    }
    m -= 1023; /* unbias exponent */
    ix0 = (ix0 & 0x000fffff) | 0x00100000;
    if (m & 1) == 1 {
        /* odd m, double x to make it even */
        ix0 += ix0 + ((ix1 & sign) >> 31) as i32;
        ix1 = ix1.wrapping_add(ix1);
    }
    m >>= 1; /* m = [m/2] */

    /* generate sqrt(x) bit by bit */
    ix0 += ix0 + ((ix1 & sign) >> 31) as i32;
    ix1 = ix1.wrapping_add(ix1);
    q = 0; /* [q,q1] = sqrt(x) */
    q1 = 0;
    s0 = 0;
    s1 = 0;
    r = 0x00200000; /* r = moving bit from right to left */

    while r != 0 {
        t = s0 + r as i32;
        if t <= ix0 {
            s0 = t + r as i32;
            ix0 -= t;
            q += r as i32;
        }
        ix0 += ix0 + ((ix1 & sign) >> 31) as i32;
        ix1 = ix1.wrapping_add(ix1);
        r >>= 1;
    }

    r = sign;
    while r != 0 {
        t1 = s1.wrapping_add(r);
        t = s0;
        if t < ix0 || (t == ix0 && t1 <= ix1) {
            s1 = t1.wrapping_add(r);
            if (t1 & sign) == sign && (s1 & sign) == 0 {
                s0 += 1;
            }
            ix0 -= t;
            if ix1 < t1 {
                ix0 -= 1;
            }
            ix1 = ix1.wrapping_sub(t1);
            q1 += r;
        }
        ix0 += ix0 + ((ix1 & sign) >> 31) as i32;
        ix1 = ix1.wrapping_add(ix1);
        r >>= 1;
    }

    /* use floating add to find out rounding direction */
    if (ix0 as u32 | ix1) != 0 {
        z = SoftF64(1.0).sub(TINY); /* raise inexact flag */
        if ge(z, SoftF64::ONE) {
            z = SoftF64::ONE.add(TINY);
            if q1 == 0xffffffff {
                q1 = 0;
                q += 1;
            } else if gt(z, SoftF64::ONE) {
                if q1 == 0xfffffffe {
                    q += 1;
                }
                q1 = q1.wrapping_add(2);
            } else {
                q1 += q1 & 1;
            }
        }
    }
    ix0 = (q >> 1) + 0x3fe00000;
    ix1 = q1 >> 1;
    if (q & 1) == 1 {
        ix1 |= sign;
    }
    ix0 += m << 20;
    SoftF64::from_bits((ix0 as u64) << 32 | ix1 as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    //  use core::f64::*;

    #[test]
    fn sanity_check() {
        const SQRT_100: SoftF64 = sqrt(SoftF64(100.0));
        assert_eq!(SQRT_100.0, 10.0);

        const SQRT_4: SoftF64 = sqrt(SoftF64(4.0));
        assert_eq!(SQRT_4.0, 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/sqrt
    #[test]
    fn spec_tests() {
        // Not Asserted: FE_INVALID exception is raised if argument is negative.
        assert!(sqrt(SoftF64(-1.0)).0.is_nan());
        assert!(sqrt(SoftF64(f64::NAN)).0.is_nan());
        for f in [0.0, -0.0, f64::INFINITY].iter().copied() {
            assert_eq!(sqrt(SoftF64(f)).0, f);
        }
    }
}
