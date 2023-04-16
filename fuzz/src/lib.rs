#![cfg(test)]
mod compiler_builtins;
mod f32_fuzz;
mod f64_fuzz;

pub enum Argument {
    First,
    Second,
}

mod sf32 {
    use super::Argument;
    use std::fmt::Display;

    use const_soft_float::soft_f32::SoftF32;

    fn is_eq(a: f32, b: f32) -> bool {
        match (a, b) {
            (a, b) if a.is_nan() && b.is_nan() => true,
            (a, b) => a == b,
        }
    }

    fn fuzz_iter() -> impl Iterator<Item = u32> {
        0..u32::MAX
    }

    pub fn fuzz_test_op(
        mut soft: impl FnMut(SoftF32) -> SoftF32,
        mut hard: impl FnMut(f32) -> f32,
        name: Option<&str>,
    ) {
        for (index, bits) in fuzz_iter().enumerate() {
            let soft_res = soft(SoftF32(f32::from_bits(bits)));
            let hard_res = hard(f32::from_bits(bits));
            if !is_eq(soft_res.0, hard_res) {
                match name {
                    Some(name) => eprintln!(
                        "NE Result for {}: \n\t Soft: {}({}) = {} \n\t Ref: {}({}) = {}",
                        name,
                        SoftF32::from_bits(bits).0,
                        name,
                        soft_res.0,
                        f32::from_bits(bits),
                        name,
                        hard_res
                    ),
                    None => eprintln!(
                        "NE Result: \n\t Soft: op({}) = {} \n\t Ref: op({}) = {}",
                        SoftF32::from_bits(bits).0,
                        soft_res.0,
                        f32::from_bits(bits),
                        hard_res
                    ),
                };
                assert_eq!(soft_res.0, hard_res)
            }

            if let (Some(name), 0) = (name, index % 10_000_00) {
                eprintln!("{}: {}", name, f32::from_bits(bits));
            }
        }
    }

    pub fn fuzz_test_op_2(
        mut soft: impl FnMut(SoftF32, SoftF32) -> SoftF32,
        mut hard: impl FnMut(f32, f32) -> f32,
        fuzz_arg: Argument,
        name: Option<&str>,
    ) {
        use nanorand::{Rng, WyRand};

        let mut soft_rng = WyRand::new_seed(WyRand::new().generate::<u64>());
        let mut hard_rng = soft_rng.clone();

        let mut soft = |x: SoftF32| -> (SoftF32, SoftF32) {
            let rand = SoftF32(f32::from_bits(soft_rng.generate::<u32>()));
            let ret = match fuzz_arg {
                Argument::First => soft(x, rand),
                Argument::Second => soft(rand, x),
            };
            (ret, rand)
        };

        let mut hard = |x: f32| -> (f32, f32) {
            let rand = f32::from_bits(hard_rng.generate::<u32>());
            let ret = match fuzz_arg {
                Argument::First => hard(x, rand),
                Argument::Second => hard(rand, x),
            };
            (ret, rand)
        };

        for (index, bits) in fuzz_iter().enumerate() {
            let soft_res = soft(SoftF32::from_bits(bits));
            let hard_res = hard(f32::from_bits(bits));
            if !is_eq(soft_res.0 .0, hard_res.0) {
                match name {
                    Some(name) => eprintln!(
                        "NE Result for {}: \n\t Soft: {} {} {} = {} \n\t Ref: {} {} {} = {}",
                        name,
                        SoftF32::from_bits(bits).0,
                        name,
                        soft_res.1 .0,
                        soft_res.0 .0,
                        f32::from_bits(bits),
                        name,
                        hard_res.1,
                        hard_res.0,
                    ),
                    None => eprintln!(
                        "NE Result: \n\t Soft: ({}, {}) -> {} \n\t Ref: ({}, {}) -> {}",
                        SoftF32::from_bits(bits).0,
                        soft_res.1 .0,
                        soft_res.0 .0,
                        f32::from_bits(bits),
                        hard_res.1,
                        hard_res.0,
                    ),
                }
                assert_eq!(soft_res.0 .0, hard_res.0)
            }

            if let (Some(name), 0) = (name, index % 10_000_00) {
                eprintln!("{}: {}", name, f32::from_bits(bits));
            }
        }
    }

    pub fn fuzz_test_op_2_other<T: nanorand::RandomGen<nanorand::WyRand, 8> + Display + Copy>(
        mut soft: impl FnMut(SoftF32, T) -> SoftF32,
        mut hard: impl FnMut(f32, T) -> f32,
        name: Option<&str>,
    ) {
        use nanorand::{Rng, WyRand};

        let mut soft_rng = WyRand::new_seed(WyRand::new().generate::<u64>());
        let mut hard_rng = soft_rng.clone();

        let mut soft = |x: SoftF32| -> (SoftF32, T) {
            let rand = soft_rng.generate::<T>();
            let ret = soft(x, rand);
            (ret, rand)
        };

        let mut hard = |x: f32| -> (f32, T) {
            let rand = hard_rng.generate::<T>();
            let ret = hard(x, rand);
            (ret, rand)
        };

        for (index, bits) in fuzz_iter().enumerate() {
            let soft_res = soft(SoftF32::from_bits(bits));
            let hard_res = hard(f32::from_bits(bits));
            if !is_eq(soft_res.0 .0, hard_res.0) {
                match name {
                    Some(name) => eprintln!(
                        "NE Result for {}: \n\t Soft: {} {} {} = {} \n\t Ref: {} {} {} = {}",
                        name,
                        SoftF32::from_bits(bits).0,
                        name,
                        soft_res.1,
                        soft_res.0 .0,
                        f32::from_bits(bits),
                        name,
                        hard_res.1,
                        hard_res.0,
                    ),
                    None => eprintln!(
                        "NE Result: \n\t Soft: ({}, {}) -> {} \n\t Ref: ({}, {}) -> {}",
                        SoftF32::from_bits(bits).0,
                        soft_res.1,
                        soft_res.0 .0,
                        f32::from_bits(bits),
                        hard_res.1,
                        hard_res.0,
                    ),
                }
                assert_eq!(soft_res.0 .0, hard_res.0)
            }

            if let (Some(name), 0) = (name, index % 10_000_00) {
                eprintln!("{}: {}", name, f32::from_bits(bits));
            }
        }
    }
}

mod sf64 {
    use std::fmt::Display;

    use const_soft_float::soft_f64::SoftF64;

    use crate::Argument;

    fn is_eq(a: f64, b: f64) -> bool {
        match (a, b) {
            (a, b) if a.is_nan() && b.is_nan() => true,
            (a, b) => a == b,
        }
    }

    fn fuzz_iter() -> impl Iterator<Item = u64> {
        let step =
            2_usize.pow(((std::mem::size_of::<f64>() - std::mem::size_of::<f32>()) * 8) as u32);
        (0..10_000_00)
            .chain((0..u64::MAX).step_by(step))
            .chain(std::iter::once(SoftF64(0.0).to_bits()))
            .chain(std::iter::once(SoftF64(-0.0).to_bits()))
            .chain((u64::MAX - 10_000_00)..10_000_00)
    }

    pub fn fuzz_test_op(
        mut soft: impl FnMut(SoftF64) -> SoftF64,
        mut hard: impl FnMut(f64) -> f64,
        name: Option<&str>,
    ) {
        for (index, bits) in fuzz_iter().enumerate() {
            let soft_res = soft(SoftF64(f64::from_bits(bits)));
            let hard_res = hard(f64::from_bits(bits));
            if !is_eq(soft_res.0, hard_res) {
                match name {
                    Some(name) => eprintln!(
                        "NE Result for {}: \n\t Soft: {}({}) = {} \n\t Ref: {}({}) = {}",
                        name,
                        SoftF64::from_bits(bits).0,
                        name,
                        soft_res.0,
                        f64::from_bits(bits),
                        name,
                        hard_res
                    ),
                    None => eprintln!(
                        "NE Result: \n\t Soft: op({}) = {} \n\t Ref: op({}) = {}",
                        SoftF64::from_bits(bits).0,
                        soft_res.0,
                        f64::from_bits(bits),
                        hard_res
                    ),
                };
                assert_eq!(soft_res.0, hard_res)
            }

            if let (Some(name), 0) = (name, index % 10_000_00) {
                eprintln!("{}: {}", name, f64::from_bits(bits));
            }
        }
    }

    pub fn fuzz_test_op_2(
        mut soft: impl FnMut(SoftF64, SoftF64) -> SoftF64,
        mut hard: impl FnMut(f64, f64) -> f64,
        fuzz_arg: Argument,
        name: Option<&str>,
    ) {
        use nanorand::{Rng, WyRand};

        let mut soft_rng = WyRand::new_seed(WyRand::new().generate::<u64>());
        let mut hard_rng = soft_rng.clone();

        let mut soft = |x: SoftF64| -> (SoftF64, SoftF64) {
            let rand = SoftF64(f64::from_bits(soft_rng.generate::<u64>()));
            let ret = match fuzz_arg {
                Argument::First => soft(x, rand),
                Argument::Second => soft(rand, x),
            };
            (ret, rand)
        };

        let mut hard = |x: f64| -> (f64, f64) {
            let rand = f64::from_bits(hard_rng.generate::<u64>());
            let ret = match fuzz_arg {
                Argument::First => hard(x, rand),
                Argument::Second => hard(rand, x),
            };
            (ret, rand)
        };

        for (index, bits) in fuzz_iter().enumerate() {
            let soft_res = soft(SoftF64::from_bits(bits));
            let hard_res = hard(f64::from_bits(bits));
            if !is_eq(soft_res.0 .0, hard_res.0) {
                match name {
                    Some(name) => eprintln!(
                        "NE Result for {}: \n\t Soft: {} {} {} = {} \n\t Ref: {} {} {} = {}",
                        name,
                        SoftF64::from_bits(bits).0,
                        name,
                        soft_res.1 .0,
                        soft_res.0 .0,
                        f64::from_bits(bits),
                        name,
                        hard_res.1,
                        hard_res.0,
                    ),
                    None => eprintln!(
                        "NE Result: \n\t Soft: ({}, {}) -> {} \n\t Ref: ({}, {}) -> {}",
                        SoftF64::from_bits(bits).0,
                        soft_res.1 .0,
                        soft_res.0 .0,
                        f64::from_bits(bits),
                        hard_res.1,
                        hard_res.0,
                    ),
                }
                assert_eq!(soft_res.0 .0, hard_res.0)
            }

            if let (Some(name), 0) = (name, index % 10_000_00) {
                eprintln!("{}: {}", name, f64::from_bits(bits));
            }
        }
    }

    pub fn fuzz_test_op_2_other<T: nanorand::RandomGen<nanorand::WyRand, 8> + Display + Copy>(
        mut soft: impl FnMut(SoftF64, T) -> SoftF64,
        mut hard: impl FnMut(f64, T) -> f64,
        name: Option<&str>,
    ) {
        use nanorand::{Rng, WyRand};

        let mut soft_rng = WyRand::new_seed(WyRand::new().generate::<u64>());
        let mut hard_rng = soft_rng.clone();

        let mut soft = |x: SoftF64| -> (SoftF64, T) {
            let rand = soft_rng.generate::<T>();
            let ret = soft(x, rand);
            (ret, rand)
        };

        let mut hard = |x: f64| -> (f64, T) {
            let rand = hard_rng.generate::<T>();
            let ret = hard(x, rand);
            (ret, rand)
        };

        for (index, bits) in fuzz_iter().enumerate() {
            let soft_res = soft(SoftF64::from_bits(bits));
            let hard_res = hard(f64::from_bits(bits));
            if !is_eq(soft_res.0 .0, hard_res.0) {
                match name {
                    Some(name) => eprintln!(
                        "NE Result for {}: \n\t Soft: {} {} {} = {} \n\t Ref: {} {} {} = {}",
                        name,
                        SoftF64::from_bits(bits).0,
                        name,
                        soft_res.1,
                        soft_res.0 .0,
                        f64::from_bits(bits),
                        name,
                        hard_res.1,
                        hard_res.0,
                    ),
                    None => eprintln!(
                        "NE Result: \n\t Soft: ({}, {}) -> {} \n\t Ref: ({}, {}) -> {}",
                        SoftF64::from_bits(bits).0,
                        soft_res.1,
                        soft_res.0 .0,
                        f64::from_bits(bits),
                        hard_res.1,
                        hard_res.0,
                    ),
                }
                assert_eq!(soft_res.0 .0, hard_res.0)
            }

            if let (Some(name), 0) = (name, index % 10_000_00) {
                eprintln!("{}: {}", name, f64::from_bits(bits));
            }
        }
    }
}
