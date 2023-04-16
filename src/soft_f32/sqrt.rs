/* origin: Rust libm https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/sqrtf.rs */
/* origin: FreeBSD /usr/src/lib/msun/src/e_sqrtf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */

use crate::soft_f32::SoftF32;
use core::cmp::Ordering;

pub(crate) const fn sqrtf(x: SoftF32) -> SoftF32 {
    const TINY: SoftF32 = SoftF32(1.0e-30);

    let mut z: SoftF32;
    let sign: i32 = 0x80000000_u32 as i32;
    let mut ix: i32;
    let mut s: i32;
    let mut q: i32;
    let mut m: i32;
    let mut t: i32;
    let mut i: i32;
    let mut r: u32;

    ix = x.to_bits() as i32;

    /* take care of Inf and NaN */
    if (ix as u32 & 0x7f800000) == 0x7f800000 {
        return x.mul(x).add(x); /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
    }

    /* take care of zero */
    if ix <= 0 {
        if (ix & !sign) == 0 {
            return x; /* sqrt(+-0) = +-0 */
        }
        if ix < 0 {
            return (x.sub(x)).div(x.sub(x)); /* sqrt(-ve) = sNaN */
        }
    }

    /* normalize x */
    m = ix >> 23;
    if m == 0 {
        /* subnormal x */
        i = 0;
        while ix & 0x00800000 == 0 {
            ix <<= 1;
            i = i + 1;
        }
        m -= i - 1;
    }
    m -= 127; /* unbias exponent */
    ix = (ix & 0x007fffff) | 0x00800000;
    if m & 1 == 1 {
        /* odd m, double x to make it even */
        ix += ix;
    }
    m >>= 1; /* m = [m/2] */

    /* generate sqrt(x) bit by bit */
    ix += ix;
    q = 0;
    s = 0;
    r = 0x01000000; /* r = moving bit from right to left */

    while r != 0 {
        t = s + r as i32;
        if t <= ix {
            s = t + r as i32;
            ix -= t;
            q += r as i32;
        }
        ix += ix;
        r >>= 1;
    }

    /* use floating add to find out rounding direction */
    if ix != 0 {
        z = SoftF32(1.0).sub(TINY); /* raise inexact flag */
        if ge(z, 1.0) {
            z = SoftF32(1.0).add(TINY);
            if gt(z, 1.0) {
                q += 2;
            } else {
                q += q & 1;
            }
        }
    }

    ix = (q >> 1) + 0x3f000000;
    ix += m << 23;
    SoftF32::from_bits(ix as u32)
}

const fn gt(l: SoftF32, r: f32) -> bool {
    if let Some(ord) = l.cmp(SoftF32(r)) {
        match ord {
            Ordering::Greater => true,
            _ => false,
        }
    } else {
        panic!("Failed to compare values");
    }
}

const fn ge(l: SoftF32, r: f32) -> bool {
    if let Some(ord) = l.cmp(SoftF32(r)) {
        match ord {
            Ordering::Less => false,
            _ => true,
        }
    } else {
        panic!("Failed to compare values");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::*;

    #[test]
    fn sanity_check() {
        assert_eq!(sqrtf(SoftF32(100.0)).0, 10.0);
        assert_eq!(sqrtf(SoftF32(4.0)).0, 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/sqrt
    #[test]
    fn spec_tests() {
        // Not Asserted: FE_INVALID exception is raised if argument is negative.
        assert!(sqrtf(SoftF32(-1.0)).0.is_nan());
        assert!(sqrtf(SoftF32(NAN)).0.is_nan());
        for f in [0.0, -0.0, INFINITY].iter().copied() {
            assert_eq!(sqrtf(SoftF32(f)).0, f);
        }
    }
}
