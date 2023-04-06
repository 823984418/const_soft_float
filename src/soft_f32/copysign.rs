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

    #[ignore]
    #[test]
    fn fuzz() {
        // fuzz base
        // pos
        {
            let y = SoftF32(1.0);
            let soft_op = |x: SoftF32| -> SoftF32 { x.copysign(y) };
            let hard_op = |x: f32| -> f32 { x.copysign(y.0) };
            SoftF32::fuzz_test_op(soft_op, hard_op, Some("neg_copysign"))
        }
        // neg
        {
            let y = SoftF32(-1.0);
            let soft_op = |x: SoftF32| -> SoftF32 { x.copysign(y) };
            let hard_op = |x: f32| -> f32 { x.copysign(y.0) };
            SoftF32::fuzz_test_op(soft_op, hard_op, Some("pos_copysign"))
        }

        // fuzz sign
        // pos
        {
            let x = SoftF32(1.0);
            let soft_op = |y: SoftF32| -> SoftF32 { x.copysign(y) };
            let hard_op = |y: f32| -> f32 { x.0.copysign(y) };
            SoftF32::fuzz_test_op(soft_op, hard_op, Some("neg_copysign_base"))
        }
        // neg
        {
            let x = SoftF32(1.0);
            let soft_op = |y: SoftF32| -> SoftF32 { x.copysign(y) };
            let hard_op = |y: f32| -> f32 { x.0.copysign(y) };
            SoftF32::fuzz_test_op(soft_op, hard_op, Some("pos_copysign_base"))
        }
    }
}
