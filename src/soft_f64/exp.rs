// origin: SoftF64reeBSD /usr/src/lib/msun/src/e_exp.c
// https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/exp.rs
//
// ====================================================
// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
//
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

use super::{
    helpers::{gt, lt, scalbn},
    SoftF64,
};

// exp(x)
// Returns the exponential of x.
//
// Method
//   1. Argument reduction:
//      Reduce x to an r so that |r| <= 0.5*ln2 ~ 0.34658.
//      Given x, find r and integer k such that
//
//               x = k*ln2 + r,  |r| <= 0.5*ln2.
//
//      Here r will be represented as r = hi-lo for better
//      accuracy.
//
//   2. Approximation of exp(r) by a special rational function on
//      the interval [0,0.34658]:
//      Write
//          R(r**2) = r*(exp(r)+1)/(exp(r)-1) = 2 + r*r/6 - r**4/360 + ...
//      We use a special Remez algorithm on [0,0.34658] to generate
//      a polynomial of degree 5 to approximate R. The maximum error
//      of this polynomial approximation is bounded by 2**-59. In
//      other words,
//          R(z) ~ 2.0 + P1*z + P2*z**2 + P3*z**3 + P4*z**4 + P5*z**5
//      (where z=r*r, and the values of P1 to P5 are listed below)
//      and
//          |                  5          |     -59
//          | 2.0+P1*z+...+P5*z   -  R(z) | <= 2
//          |                             |
//      The computation of exp(r) thus becomes
//                              2*r
//              exp(r) = 1 + ----------
//                            R(r) - r
//                                 r*c(r)
//                     = 1 + r + ----------- (for better accuracy)
//                                2 - c(r)
//      where
//                              2       4             10
//              c(r) = r - (P1*r  + P2*r  + ... + P5*r   ).
//
//   3. Scale back to obtain exp(x):
//      From step 1, we have
//         exp(x) = 2^k * exp(r)
//
// Special cases:
//      exp(INF) is INF, exp(NaN) is NaN;
//      exp(-INF) is 0, and
//      for finite argument, only exp(0)=1 is exact.
//
// Accuracy:
//      according to an error analysis, the error is always less than
//      1 ulp (unit in the last place).
//
// Misc. info.
//      For IEEE double
//          if x >  709.782712893383973096 then exp(x) overflows
//          if x < -745.133219101941108420 then exp(x) underflows

const HALF: [SoftF64; 2] = [SoftF64(0.5), SoftF64(-0.5)];
const LN2HI: SoftF64 = SoftF64(6.93147180369123816490e-01); /* 0x3fe62e42, 0xfee00000 */
const LN2LO: SoftF64 = SoftF64(1.90821492927058770002e-10); /* 0x3dea39ef, 0x35793c76 */
const INVLN2: SoftF64 = SoftF64(1.44269504088896338700e+00); /* 0x3ff71547, 0x652b82fe */
const P1: SoftF64 = SoftF64(1.66666666666666019037e-01); /* 0x3FC55555, 0x5555553E */
const P2: SoftF64 = SoftF64(-2.77777777770155933842e-03); /* 0xBF66C16C, 0x16BEBD93 */
const P3: SoftF64 = SoftF64(6.61375632143793436117e-05); /* 0x3F11566A, 0xAF25DE2C */
const P4: SoftF64 = SoftF64(-1.65339022054652515390e-06); /* 0xBEBBBD41, 0xC5D26BF1 */
const P5: SoftF64 = SoftF64(4.13813679705723846039e-08); /* 0x3E663769, 0x72BEA4D0 */

/// Exponential, base *e* (f64)
///
/// Calculate the exponential of `x`, that is, *e* raised to the power `x`
/// (where *e* is the base of the natural system of logarithms, approximately 2.71828).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub(crate) const fn exp(mut x: SoftF64) -> SoftF64 {
    let x1p1023 = SoftF64::from_bits(0x7fe0000000000000); // 0x1p1023 === 2 ^ 1023
                                                          // let x1p_149 = SoftF64::from_bits(0x36a0000000000000); // 0x1p-149 === 2 ^ -149

    let hi: SoftF64;
    let lo: SoftF64;
    let c: SoftF64;
    let xx: SoftF64;
    let y: SoftF64;
    let k: i32;
    let sign: i32;
    let mut hx: u32;

    let abs_mask = SoftF64::SIGN_MASK - 1;

    let x_bits = x.to_bits();

    hx = (x_bits >> 32) as u32;
    sign = (hx >> 31) as i32;
    hx &= 0x7fffffff; /* high word of |x| */

    /* special cases */
    if hx >= 0x4086232b {
        /* if |x| >= 708.39... */
        if x_bits & abs_mask > SoftF64::EXPONENT_MASK {
            return x;
        }
        if gt(x, SoftF64(709.782712893383973096)) {
            /* overflow if x!=inf */
            x = x.mul(x1p1023);
            return x;
        }
        if lt(x, SoftF64(-708.39641853226410622)) {
            /* underflow if x!=-inf */
            if lt(x, SoftF64(-745.13321910194110842)) {
                return SoftF64(0.);
            }
        }
    }

    /* argument reduction */
    if hx > 0x3fd62e42 {
        /* if |x| > 0.5 ln2 */
        if hx >= 0x3ff0a2b2 {
            /* if |x| >= 1.5 ln2 */
            k = INVLN2.mul(x).add(HALF[sign as usize]).0 as i32;
        } else {
            k = 1 - sign - sign;
        }
        hi = x.sub(SoftF64(k as f64).mul(LN2HI)); /* k*ln2hi is exact here */
        lo = SoftF64(k as f64).mul(LN2LO);
        x = hi.sub(lo);
    } else if hx > 0x3e300000 {
        /* if |x| > 2**-28 */
        k = 0;
        hi = x;
        lo = SoftF64(0.);
    } else {
        /* inexact if x!=0 */
        return SoftF64(1.).add(x);
    }

    /* x is now in primary range */
    xx = x.mul(x);
    c = x.sub(xx.mul(P1.add(xx.mul(P2.add(xx.mul(P3.add(xx.mul(P4.add(xx.mul(P5))))))))));
    y = SoftF64(1.).add(x.mul(c).div(SoftF64(2.).sub(c)).sub(lo).add(hi));
    if k == 0 {
        y
    } else {
        scalbn(y, k)
    }
}
