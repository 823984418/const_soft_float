use super::{
    helpers::{eq, gt},
    SoftF64,
};

const TOINT: SoftF64 = SoftF64(1.0).div(SoftF64(f64::EPSILON));

/// Floor (f64)
///
/// Finds the nearest integer less than or equal to `x`.
pub const fn floor(x: SoftF64) -> SoftF64 {
    let ui = x.to_bits();
    let e = ((ui >> 52) & 0x7ff) as i32;

    if (e >= 0x3ff + 52) || eq(x, SoftF64::ZERO) {
        return x;
    }
    /* y = int(x) - x, where int(x) is an integer neighbor of x */
    let y = if (ui >> 63) != 0 {
        x.sub(TOINT).add(TOINT).sub(x)
    } else {
        x.add(TOINT).sub(TOINT).sub(x)
    };
    /* special case because of non-nearest rounding modes */
    if e < 0x3ff {
        return if (ui >> 63) != 0 {
            SoftF64(-1.0)
        } else {
            SoftF64::ZERO
        };
    }
    if gt(y, SoftF64::ZERO) {
        x.add(y).sub(SoftF64::ONE)
    } else {
        x.add(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_check() {
        assert_eq!(floor(SoftF64(1.1)).0, 1.0);
        assert_eq!(floor(SoftF64(2.9)).0, 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/floor
    #[test]
    fn spec_tests() {
        // Not Asserted: that the current rounding mode has no effect.
        assert!(floor(SoftF64(f64::NAN)).0.is_nan());
        for f in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]
            .iter()
            .copied()
        {
            assert_eq!(floor(SoftF64(f)).0, f);
        }
    }
}
