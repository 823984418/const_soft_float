use super::SoftF32;

pub(crate) const fn round(x: SoftF32) -> SoftF32 {
    SoftF32::trunc(x.add(SoftF32::copysign(
        SoftF32(0.5).sub(SoftF32(0.25).mul(SoftF32(f32::EPSILON))),
        x,
    )))
}

#[cfg(test)]
mod tests {
    use super::SoftF32;

    #[test]
    fn negative_zero() {
        assert_eq!(
            SoftF32::round(SoftF32(-0.0)).to_bits(),
            SoftF32(-0.0).to_bits()
        );
    }

    #[test]
    fn sanity_check() {
        assert_eq!((SoftF32(-1.0)).round().0, -1.0);
        assert_eq!((SoftF32(2.8)).round().0, 3.0);
        assert_eq!((SoftF32(-0.5)).round().0, -1.0);
        assert_eq!((SoftF32(0.5)).round().0, 1.0);
        assert_eq!((SoftF32(-1.5)).round().0, -2.0);
        assert_eq!((SoftF32(1.5)).round().0, 2.0);
    }
}
