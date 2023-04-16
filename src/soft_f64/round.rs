use super::SoftF64;

pub(crate) const fn round(x: SoftF64) -> SoftF64 {
    SoftF64::trunc(x.add(SoftF64::copysign(
        SoftF64(0.5).sub(SoftF64(0.25).mul(SoftF64(f64::EPSILON))),
        x,
    )))
}

#[cfg(test)]
mod tests {
    use super::SoftF64;

    #[test]
    fn negative_zero() {
        assert_eq!(
            SoftF64::round(SoftF64(-0.0)).to_bits(),
            SoftF64(-0.0).to_bits()
        );
    }

    #[test]
    fn sanity_check() {
        assert_eq!((SoftF64(-1.0)).round().0, -1.0);
        assert_eq!((SoftF64(2.8)).round().0, 3.0);
        assert_eq!((SoftF64(-0.5)).round().0, -1.0);
        assert_eq!((SoftF64(0.5)).round().0, 1.0);
        assert_eq!((SoftF64(-1.5)).round().0, -2.0);
        assert_eq!((SoftF64(1.5)).round().0, 2.0);
    }
}
