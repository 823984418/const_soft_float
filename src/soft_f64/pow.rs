use crate::abs_diff;
use crate::soft_f64::SoftF64;

type F = SoftF64;

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
    use crate::soft_f64::SoftF64;

    #[test]
    fn sanity_check() {
        assert_eq!(SoftF64(2.0).powi(2).0, 4.0)
    }

    #[ignore]
    #[test]
    fn fuzz_pow() {
        use nanorand::{Rng, WyRand};

        let mut soft_rng = WyRand::new_seed(WyRand::new().generate::<u64>());
        let mut hard_rng = soft_rng.clone();

        let soft = |x: SoftF64| -> SoftF64 {
            let other = soft_rng.generate::<u16>() as i32;
            x.powi(other)
        };

        let hard = |x: f64| -> f64 {
            let other = hard_rng.generate::<u16>() as i32;
            x.powi(other)
        };

        SoftF64::fuzz_test_op_epsilon(soft, hard, Some("pow"))
    }
}
