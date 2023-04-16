use super::SoftF32;

pub(crate) const fn trunc(x: SoftF32) -> SoftF32 {
    let mut i: u32 = x.to_bits();
    let mut e: i32 = (i >> 23 & 0xff) as i32 - 0x7f + 9;
    let m: u32;

    if e >= 23 + 9 {
        return x;
    }
    if e < 9 {
        e = 1;
    }
    m = -1i32 as u32 >> e;
    if (i & m) == 0 {
        return x;
    }
    i &= !m;
    SoftF32::from_bits(i)
}

#[cfg(test)]
mod tests {
    use crate::soft_f32::SoftF32;

    #[test]
    fn sanity_check() {
        assert_eq!(super::trunc(SoftF32(1.1)).0, 1.0);
    }
}
