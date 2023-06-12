// origin: FreeBSD /usr/src/lib/msun/src/e_log.c
// https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/log.rs
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

use super::SoftF64;

// log(x)
// Return the logarithm of x
//
// Method :
//   1. Argument Reduction: find k and f such that
//                      x = 2^k * (1+f),
//         where  sqrt(2)/2 < 1+f < sqrt(2) .
//
//   2. Approximation of log(1+f).
//      Let s = f/(2+f) ; based on log(1+f) = log(1+s) - log(1-s)
//               = 2s + 2/3 s**3 + 2/5 s**5 + .....,
//               = 2s + s*R
//      We use a special Remez algorithm on [0,0.1716] to generate
//      a polynomial of degree 14 to approximate R The maximum error
//      of this polynomial approximation is bounded by 2**-58.45. In
//      other words,
//                      2      4      6      8      10      12      14
//          R(z) ~ Lg1*s +Lg2*s +Lg3*s +Lg4*s +Lg5*s  +Lg6*s  +Lg7*s
//      (the values of Lg1 to Lg7 are listed in the program)
//      and
//          |      2          14          |     -58.45
//          | Lg1*s +...+Lg7*s    -  R(z) | <= 2
//          |                             |
//      Note that 2s = f - s*f = f - hfsq + s*hfsq, where hfsq = f*f/2.
//      In order to guarantee error in log below 1ulp, we compute log
//      by
//              log(1+f) = f - s*(f - R)        (if f is not too large)
//              log(1+f) = f - (hfsq - s*(hfsq+R)).     (better accuracy)
//
//      3. Finally,  log(x) = k*ln2 + log(1+f).
//                          = k*ln2_hi+(f-(hfsq-(s*(hfsq+R)+k*ln2_lo)))
//         Here ln2 is split into two floating point number:
//                      ln2_hi + ln2_lo,
//         where n*ln2_hi is always exact for |n| < 2000.
//
// Special cases:
//      log(x) is NaN with signal if x < 0 (including -INF) ;
//      log(+INF) is +INF; log(0) is -INF with signal;
//      log(NaN) is that NaN with no signal.
//
// Accuracy:
//      according to an error analysis, the error is always less than
//      1 ulp (unit in the last place).
//
// Constants:
// The hexadecimal values are the intended ones for the following
// constants. The decimal values may be used, provided that the
// compiler will convert from decimal to binary accurately enough
// to produce the hexadecimal values shown.

const LN2_HI: SoftF64 = SoftF64(6.93147180369123816490e-01); /* 3fe62e42 fee00000 */
const LN2_LO: SoftF64 = SoftF64(1.90821492927058770002e-10); /* 3dea39ef 35793c76 */
const LG1: SoftF64 = SoftF64(6.666666666666735130e-01); /* 3FE55555 55555593 */
const LG2: SoftF64 = SoftF64(3.999999999940941908e-01); /* 3FD99999 9997FA04 */
const LG3: SoftF64 = SoftF64(2.857142874366239149e-01); /* 3FD24924 94229359 */
const LG4: SoftF64 = SoftF64(2.222219843214978396e-01); /* 3FCC71C5 1D8E78AF */
const LG5: SoftF64 = SoftF64(1.818357216161805012e-01); /* 3FC74664 96CB03DE */
const LG6: SoftF64 = SoftF64(1.531383769920937332e-01); /* 3FC39A09 D078C69F */
const LG7: SoftF64 = SoftF64(1.479819860511658591e-01); /* 3FC2F112 DF3E5244 */

pub(crate) const fn log(mut x: SoftF64) -> SoftF64 {
    let x1p54 = SoftF64::from_bits(0x4350000000000000); // 0x1p54 === 2 ^ 54

    let mut ui = x.to_bits();
    let mut hx: u32 = (ui >> 32) as u32;
    let mut k: i32 = 0;

    if (hx < 0x00100000) || ((hx >> 31) != 0) {
        /* x < 2**-126  */
        if ui << 1 == 0 {
            return SoftF64(-1.).div(x.mul(x)); /* log(+-0)=-inf */
        }
        if hx >> 31 != 0 {
            return (x.sub(x)).div(SoftF64::ZERO); /* log(-#) = NaN */
        }
        /* subnormal number, scale x up */
        k -= 54;
        x = x.mul(x1p54);
        ui = x.to_bits();
        hx = (ui >> 32) as u32;
    } else if hx >= 0x7ff00000 {
        return x;
    } else if hx == 0x3ff00000 && ui << 32 == 0 {
        return SoftF64::ZERO;
    }

    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    hx += 0x3ff00000 - 0x3fe6a09e;
    k += ((hx >> 20) as i32) - 0x3ff;
    hx = (hx & 0x000fffff) + 0x3fe6a09e;
    ui = ((hx as u64) << 32) | (ui & 0xffffffff);
    x = SoftF64::from_bits(ui);

    let f = x.sub(SoftF64::ONE);
    let hfsq = SoftF64(0.5).mul(f).mul(f);
    let s = f.div(SoftF64(2.0).add(f));
    let z = s.mul(s);
    let w = z.mul(z);
    let t1 = w.mul(LG2.add(w.mul(LG4.add(w.mul(LG6)))));
    let t2 = z.mul(LG1.add(w.mul(LG3.add(w.mul(LG5.add(w.mul(LG7)))))));
    let r = t2.add(t1);
    let dk = SoftF64(k as f64);
    s.mul(hfsq.add(r))
        .add(dk.mul(LN2_LO))
        .sub(hfsq)
        .add(f)
        .add(dk.mul(LN2_HI))
}
