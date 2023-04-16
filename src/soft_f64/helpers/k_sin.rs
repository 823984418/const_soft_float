// origin: FreeBSD /usr/src/lib/msun/src/k_sin.c,
// https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/k_sin.rs
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

use crate::soft_f64::SoftF64;

const S1: SoftF64 = SoftF64(-1.66666666666666324348e-01); /* 0xBFC55555, 0x55555549 */
const S2: SoftF64 = SoftF64(8.33333333332248946124e-03); /* 0x3F811111, 0x1110F8A6 */
const S3: SoftF64 = SoftF64(-1.98412698298579493134e-04); /* 0xBF2A01A0, 0x19C161D5 */
const S4: SoftF64 = SoftF64(2.75573137070700676789e-06); /* 0x3EC71DE3, 0x57B1FE7D */
const S5: SoftF64 = SoftF64(-2.50507602534068634195e-08); /* 0xBE5AE5E6, 0x8A2B9CEB */
const S6: SoftF64 = SoftF64(1.58969099521155010221e-10); /* 0x3DE5D93A, 0x5ACFD57C */

// kernel sin function on ~[-pi/4, pi/4] (except on -0), pi/4 ~ 0.7854
// Input x is assumed to be bounded by ~pi/4 in magnitude.
// Input y is the tail of x.
// Input iy indicates whether y is 0. (if iy=0, y assume to be 0).
//
// Algorithm
//      1. Since sin(-x) = -sin(x), we need only to consider positive x.
//      2. Callers must return sin(-0) = -0 without calling here since our
//         odd polynomial is not evaluated in a way that preserves -0.
//         Callers may do the optimization sin(x) ~ x for tiny x.
//      3. sin(x) is approximated by a polynomial of degree 13 on
//         [0,pi/4]
//                               3            13
//              sin(x) ~ x + S1*x + ... + S6*x
//         where
//
//      |sin(x)         2     4     6     8     10     12  |     -58
//      |----- - (1+S1*x +S2*x +S3*x +S4*x +S5*x  +S6*x   )| <= 2
//      |  x                                               |
//
//      4. sin(x+y) = sin(x) + sin'(x')*y
//                  ~ sin(x) + (1-x*x/2)*y
//         For better accuracy, let
//                   3      2      2      2      2
//              r = x *(S2+x *(S3+x *(S4+x *(S5+x *S6))))
//         then                   3    2
//              sin(x) = x + (S1*x + (x *(r-y/2)+y))
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub(crate) const fn k_sin(x: SoftF64, y: SoftF64, iy: i32) -> SoftF64 {
    let z = x.mul(x);
    let w = z.mul(z);
    let r = S2
        .add(z.mul(S3.add(z.mul(S4))))
        .add(z.mul(w.mul(S5.add(z.mul(S6)))));
    let v = z.mul(x);
    if iy == 0 {
        x.add(v.mul(S1.add(z.mul(r))))
    } else {
        x.sub((z.mul(SoftF64(0.5).mul(y).sub(v.mul(r))).sub(y)).sub(v.mul(S1)))
    }
}
