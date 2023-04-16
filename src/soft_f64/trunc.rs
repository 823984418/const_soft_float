use super::SoftF64;

pub(crate) const fn trunc(x: SoftF64) -> SoftF64 {
    let mut i: u64 = x.to_bits();
    let mut e: i64 = (i >> 52 & 0x7ff) as i64 - 0x3ff + 12;

    if e >= 52 + 12 {
        return x;
    }
    if e < 12 {
        e = 1;
    }
    let m = -1i64 as u64 >> e;
    if (i & m) == 0 {
        return x;
    }
    i &= !m;
    SoftF64::from_bits(i)
}

#[cfg(test)]
mod tests {
    use crate::soft_f64::SoftF64;

    #[test]
    fn sanity_check() {
        assert_eq!(super::trunc(SoftF64(1.1)).0, 1.0);
    }
}
