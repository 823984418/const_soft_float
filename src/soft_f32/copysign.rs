use super::SoftF32;

/// Sign of Y, magnitude of X (SoftF32)
///
/// Constructs a number with the magnitude (absolute value) of its
/// first argument, `x`, and the sign of its second argument, `y`.
pub(crate) const fn copysign(x: SoftF32, y: SoftF32) -> SoftF32 {
    let mut ux = x.to_bits();
    let uy = y.to_bits();
    ux &= 0x7fffffff;
    ux |= uy & 0x80000000;
    SoftF32::from_bits(ux)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity_check() {
        assert_eq!(SoftF32(1.0).copysign(SoftF32(-0.0)).0, -1.0)
    }
}
