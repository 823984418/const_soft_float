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

#[cfg(test)]
mod test {
    use super::*;

    #[ignore]
    #[test]
    fn fuzz() {
        // fuzz base
        // pos
        {
            let y = SoftF64(1.0);
            let soft_op = |x: SoftF64| -> SoftF64 { x.copysign(y) };
            let hard_op = |x: f64| -> f64 { x.copysign(y.0) };
            SoftF64::fuzz_test_op(soft_op, hard_op, Some("neg_copysign"))
        }
        // neg
        {
            let y = SoftF64(-1.0);
            let soft_op = |x: SoftF64| -> SoftF64 { x.copysign(y) };
            let hard_op = |x: f64| -> f64 { x.copysign(y.0) };
            SoftF64::fuzz_test_op(soft_op, hard_op, Some("pos_copysign"))
        }

        // fuzz sign
        // pos
        {
            let x = SoftF64(1.0);
            let soft_op = |y: SoftF64| -> SoftF64 { x.copysign(y) };
            let hard_op = |y: f64| -> f64 { x.0.copysign(y) };
            SoftF64::fuzz_test_op(soft_op, hard_op, Some("neg_copysign_base"))
        }
        // neg
        {
            let x = SoftF64(1.0);
            let soft_op = |y: SoftF64| -> SoftF64 { x.copysign(y) };
            let hard_op = |y: f64| -> f64 { x.0.copysign(y) };
            SoftF64::fuzz_test_op(soft_op, hard_op, Some("pos_copysign_base"))
        }
    }
}
