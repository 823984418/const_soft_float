use crate::soft_f64::{helpers::eq, SoftF64};

// origin: FreeBSD /usr/src/lib/msun/src/e_rem_pio2.c
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Optimized by Bruce D. Evans. */
use super::rem_pio2_large;

// #if FLT_EVAL_METHOD==0 || FLT_EVAL_METHOD==1
// #define EPS DBL_EPSILON
const EPS: SoftF64 = SoftF64(2.2204460492503131e-16);
// #elif FLT_EVAL_METHOD==2
// #define EPS LDBL_EPSILON
// #endif

// TODO: Support FLT_EVAL_METHOD?

const TO_INT: SoftF64 = SoftF64(1.5).div(EPS);
/// 53 bits of 2/pi
const INV_PIO2: SoftF64 = SoftF64(6.36619772367581382433e-01); /* 0x3FE45F30, 0x6DC9C883 */
/// first 33 bits of pi/2
const PIO2_1: SoftF64 = SoftF64(1.57079632673412561417e+00); /* 0x3FF921FB, 0x54400000 */
/// pi/2 - PIO2_1
const PIO2_1T: SoftF64 = SoftF64(6.07710050650619224932e-11); /* 0x3DD0B461, 0x1A626331 */
/// second 33 bits of pi/2
const PIO2_2: SoftF64 = SoftF64(6.07710050630396597660e-11); /* 0x3DD0B461, 0x1A600000 */
/// pi/2 - (PIO2_1+PIO2_2)
const PIO2_2T: SoftF64 = SoftF64(2.02226624879595063154e-21); /* 0x3BA3198A, 0x2E037073 */
/// third 33 bits of pi/2
const PIO2_3: SoftF64 = SoftF64(2.02226624871116645580e-21); /* 0x3BA3198A, 0x2E000000 */
/// pi/2 - (PIO2_1+PIO2_2+PIO2_3)
const PIO2_3T: SoftF64 = SoftF64(8.47842766036889956997e-32); /* 0x397B839A, 0x252049C1 */

// return the remainder of x rem pi/2 in y[0]+y[1]
// use rem_pio2_large() for large x
//
// caller must handle the case when reduction is not needed: |x| ~<= pi/4 */
pub(crate) const fn rem_pio2(x: SoftF64) -> (i32, SoftF64, SoftF64) {
    let x1p24 = SoftF64::from_bits(0x4170000000000000);

    let sign = (SoftF64::to_bits(x) >> 63) as i32;
    let ix = (SoftF64::to_bits(x) >> 32) as u32 & 0x7fffffff;

    const fn medium(x: SoftF64, ix: u32) -> (i32, SoftF64, SoftF64) {
        /* rint(x/(pi/2)), Assume round-to-nearest. */
        let tmp = SoftF64(x.0 as f64).mul(INV_PIO2).add(TO_INT);
        // force rounding of tmp to it's storage format on x87 to avoid
        // excess precision issues.
        let f_n = tmp.sub(TO_INT);
        let n = f_n.0 as i32;
        let mut r = x.sub(f_n.mul(PIO2_1));
        let mut w = f_n.mul(PIO2_1T); /* 1st round, good to 85 bits */
        let mut y0 = r.sub(w);
        let ui = SoftF64::to_bits(y0);
        let ey = (ui >> 52) as i32 & 0x7ff;
        let ex = (ix >> 20) as i32;
        if ex - ey > 16 {
            /* 2nd round, good to 118 bits */
            let t = r;
            w = f_n.mul(PIO2_2);
            r = t.sub(w);
            w = f_n.mul(PIO2_2T).sub((t.sub(r)).sub(w));
            y0 = r.sub(w);
            let ey = (SoftF64::to_bits(y0) >> 52) as i32 & 0x7ff;
            if ex - ey > 49 {
                /* 3rd round, good to 151 bits, covers all cases */
                let t = r;
                w = f_n.mul(PIO2_3);
                r = t.sub(w);
                w = f_n.mul(PIO2_3T).sub((t.sub(r)).sub(w));
                y0 = r.sub(w);
            }
        }
        let y1 = (r.sub(y0)).sub(w);
        (n, y0, y1)
    }

    if ix <= 0x400f6a7a {
        /* |x| ~<= 5pi/4 */
        if (ix & 0xfffff) == 0x921fb {
            /* |x| ~= pi/2 or 2pi/2 */
            return medium(x, ix); /* cancellation -- use medium case */
        }
        if ix <= 0x4002d97c {
            /* |x| ~<= 3pi/4 */
            if sign == 0 {
                let z = x.sub(PIO2_1); /* one round good to 85 bits */
                let y0 = z.sub(PIO2_1T);
                let y1 = (z.sub(y0)).sub(PIO2_1T);
                return (1, y0, y1);
            } else {
                let z = x.add(PIO2_1);
                let y0 = z.add(PIO2_1T);
                let y1 = (z.sub(y0)).add(PIO2_1T);
                return (-1, y0, y1);
            }
        } else if sign == 0 {
            let z = x.sub(SoftF64(2.0).mul(PIO2_1));
            let y0 = z.sub(SoftF64(2.0).mul(PIO2_1T));
            let y1 = (z.sub(y0)).sub(SoftF64(2.0).mul(PIO2_1T));
            return (2, y0, y1);
        } else {
            let z = x.add(SoftF64(2.0).mul(PIO2_1));
            let y0 = z.add(SoftF64(2.0).mul(PIO2_1T));
            let y1 = (z.sub(y0)).add(SoftF64(2.0).mul(PIO2_1T));
            return (-2, y0, y1);
        }
    }
    if ix <= 0x401c463b {
        /* |x| ~<= 9pi/4 */
        if ix <= 0x4015fdbc {
            /* |x| ~<= 7pi/4 */
            if ix == 0x4012d97c {
                /* |x| ~= 3pi/2 */
                return medium(x, ix);
            }
            if sign == 0 {
                let z = x.sub(SoftF64(3.0).mul(PIO2_1));
                let y0 = z.sub(SoftF64(3.0).mul(PIO2_1T));
                let y1 = (z.sub(y0)).sub(SoftF64(3.0).mul(PIO2_1T));
                return (3, y0, y1);
            } else {
                let z = x.add(SoftF64(3.0).mul(PIO2_1));
                let y0 = z.add(SoftF64(3.0).mul(PIO2_1T));
                let y1 = (z.sub(y0)).add(SoftF64(3.0).mul(PIO2_1T));
                return (-3, y0, y1);
            }
        } else {
            if ix == 0x401921fb {
                /* |x| ~= 4pi/2 */
                return medium(x, ix);
            }
            if sign == 0 {
                let z = x.sub(SoftF64(4.0).mul(PIO2_1));
                let y0 = z.sub(SoftF64(4.0).mul(PIO2_1T));
                let y1 = (z.sub(y0)).sub(SoftF64(4.0).mul(PIO2_1T));
                return (4, y0, y1);
            } else {
                let z = x.add(SoftF64(4.0).mul(PIO2_1));
                let y0 = z.add(SoftF64(4.0).mul(PIO2_1T));
                let y1 = (z.sub(y0)).add(SoftF64(4.0).mul(PIO2_1T));
                return (-4, y0, y1);
            }
        }
    }
    if ix < 0x413921fb {
        /* |x| ~< 2^20*(pi/2), medium size */
        return medium(x, ix);
    }
    /*
     * all other (large) arguments
     */
    if ix >= 0x7ff00000 {
        /* x is inf or NaN */
        let y0 = x.sub(x);
        let y1 = y0;
        return (0, y0, y1);
    }
    /* set z = scalbn(|x|,-ilogb(x)+23) */
    let mut ui = SoftF64::to_bits(x);
    ui &= (!1) >> 12;
    ui |= (0x3ff + 23) << 52;
    let mut z = SoftF64::from_bits(ui);
    let mut tx = [SoftF64::ZERO; 3];
    {
        let mut i = 0;
        while i < 2 {
            tx[i] = SoftF64(z.0 as i32 as f64);
            z = (z.sub(tx[i])).mul(x1p24);
            i += 1;
        }
    }
    tx[2] = z;
    /* skip zero terms, first term is non-zero */
    let mut i = 2;
    while i != 0 && eq(tx[i], SoftF64::ZERO) {
        i -= 1;
    }
    let ty = [SoftF64::ZERO; 3];
    let (n, ty) = match i {
        2 => rem_pio2_large(&tx, &ty, ((ix as i32) >> 20) - (0x3ff + 23), 1),
        1 => rem_pio2_large(&[tx[0], tx[1]], &ty, ((ix as i32) >> 20) - (0x3ff + 23), 1),
        0 => rem_pio2_large(&[tx[0]], &ty, ((ix as i32) >> 20) - (0x3ff + 23), 1),
        _ => panic!(),
    };
    if sign != 0 {
        return (-n, ty[0].neg(), ty[1].neg());
    }
    (n, ty[0], ty[1])
}

#[cfg(test)]
mod tests {
    use super::{rem_pio2, SoftF64};

    #[test]
    fn test_near_pi() {
        let arg = SoftF64(3.141592025756836);
        let (a, b, c) = rem_pio2(arg);
        assert_eq!(
            (a, b.0, c.0),
            (2, -6.278329573009626e-7, -2.1125998133974653e-23)
        );
        let arg = SoftF64(3.141592033207416);
        let (a, b, c) = rem_pio2(arg);
        assert_eq!(
            (a, b.0, c.0),
            (2, -6.20382377148128e-7, -2.1125998133974653e-23)
        );
        let arg = SoftF64(3.141592144966125);
        let (a, b, c) = rem_pio2(arg);
        assert_eq!(
            (a, b.0, c.0),
            (2, -5.086236681942706e-7, -2.1125998133974653e-23)
        );
        let arg = SoftF64(3.141592979431152);
        let (a, b, c) = rem_pio2(arg);
        assert_eq!(
            (a, b.0, c.0),
            (2, 3.2584135866119817e-7, -2.1125998133974653e-23)
        );
    }

    #[test]
    fn test_overflow_b9b847() {
        let _ = rem_pio2(SoftF64(-3054214.5490637687));
    }

    #[test]
    fn test_overflow_4747b9() {
        let _ = rem_pio2(SoftF64(917340800458.2274));
    }
}
