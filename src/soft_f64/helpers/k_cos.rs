// origin: FreeBSD /usr/src/lib/msun/src/k_cos.c,
// https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/k_cos.rs
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

const C1: SoftF64 = SoftF64(4.16666666666666019037e-02); /* 0x3FA55555, 0x5555554C */
const C2: SoftF64 = SoftF64(-1.38888888888741095749e-03); /* 0xBF56C16C, 0x16C15177 */
const C3: SoftF64 = SoftF64(2.48015872894767294178e-05); /* 0x3EFA01A0, 0x19CB1590 */
const C4: SoftF64 = SoftF64(-2.75573143513906633035e-07); /* 0xBE927E4F, 0x809C52AD */
const C5: SoftF64 = SoftF64(2.08757232129817482790e-09); /* 0x3E21EE9E, 0xBDB4B1C4 */
const C6: SoftF64 = SoftF64(-1.13596475577881948265e-11); /* 0xBDA8FAE9, 0xBE8838D4 */

// kernel cos function on [-pi/4, pi/4], pi/4 ~ 0.785398164
// Input x is assumed to be bounded by ~pi/4 in magnitude.
// Input y is the tail of x.
//
// Algorithm
//      1. Since cos(-x) = cos(x), we need only to consider positive x.
//      2. if x < 2^-27 (hx<0x3e400000 0), return 1 with inexact if x!=0.
//      3. cos(x) is approximated by a polynomial of degree 14 on
//         [0,pi/4]
//                                       4            14
//              cos(x) ~ 1 - x*x/2 + C1*x + ... + C6*x
//         where the remez error is
//
//      |              2     4     6     8     10    12     14 |     -58
//      |cos(x)-(1-.5*x +C1*x +C2*x +C3*x +C4*x +C5*x  +C6*x  )| <= 2
//      |                                                      |
//
//                     4     6     8     10    12     14
//      4. let r = C1*x +C2*x +C3*x +C4*x +C5*x  +C6*x  , then
//             cos(x) ~ 1 - x*x/2 + r
//         since cos(x+y) ~ cos(x) - sin(x)*y
//                        ~ cos(x) - x*y,
//         a correction term is necessary in cos(x) and hence
//              cos(x+y) = 1 - (x*x/2 - (r - x*y))
//         For better accuracy, rearrange to
//              cos(x+y) ~ w + (tmp + (r-x*y))
//         where w = 1 - x*x/2 and tmp is a tiny correction term
//         (1 - x*x/2 == w + tmp exactly in infinite precision).
//         The exactness of w + tmp in infinite precision depends on w
//         and tmp having the same precision as x.  If they have extra
//         precision due to compiler bugs, then the extra precision is
//         only good provided it is retained in all terms of the final
//         expression for cos().  Retention happens in all cases tested
//         under FreeBSD, so don't pessimize things by forcibly clipping
//         any extra precision in w.
pub(crate) const fn k_cos(x: SoftF64, y: SoftF64) -> SoftF64 {
    let z = x.mul(x);
    let w = z.mul(z);
    let r = z
        .mul(C1.add(z.mul(C2.add(z.mul(C3)))))
        .add(w.mul(w.mul(C4.add(z.mul(C5.add(z.mul(C6)))))));
    let hz = SoftF64(0.5).mul(z);
    let w = SoftF64::ZERO.sub(hz);
    w.add(((SoftF64::ONE.sub(w)).sub(hz)).add(z.mul(r).sub(x.mul(y))))
}
