use crate::abs_diff;
use crate::soft_f32::SoftF32;

type F = SoftF32;

pub(crate) const fn pow(a: F, b: i32) -> F {
    let mut a = a;
    let recip = b < 0;
    let mut pow = abs_diff(b, 0);
    let mut mul = F::ONE;
    loop {
        if (pow & 1) != 0 {
            mul = mul.mul(a);
        }
        pow >>= 1;
        if pow == 0 {
            break;
        }
        a = a.mul(a);
    }

    if recip {
        F::ONE.div(mul)
    } else {
        mul
    }
}

#[cfg(test)]
mod test {
    use crate::soft_f32::SoftF32;

    #[test]
    fn sanity_check() {
        assert_eq!(SoftF32(2.0).powi(2).0, 4.0)
    }
}
