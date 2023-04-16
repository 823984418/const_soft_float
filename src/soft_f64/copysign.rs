use super::SoftF64;

/// Sign of Y, magnitude of X (f64)
///
/// Constructs a number with the magnitude (absolute value) of its
/// first argument, `x`, and the sign of its second argument, `y`.
pub(crate) const fn copysign(x: SoftF64, y: SoftF64) -> SoftF64 {
    let mut ux = x.to_bits();
    let uy = y.to_bits();
    ux &= (!0) >> 1;
    ux |= uy & (1 << 63);
    SoftF64::from_bits(ux)
}
