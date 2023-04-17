//! # Rust float-point in constant context
//!
//! Features:
//! * `no_std`
//! * `const_trait_impl`
//! * `const_mut_refs`
//!
//! work in `stable`:
//! ```
//! # use const_soft_float::soft_f32::SoftF32;
//! const fn const_f32_add(a: f32, b: f32) -> f32 {
//!     SoftF32(a).add(SoftF32(b)).to_f32()
//! }
//! ```
//!
//!
//! with `const_trait_impl` usage (requires `nightly`):
//! ```
//! # cfg_if::cfg_if! {
//! # if #[cfg(nightly)] {
//! # #![feature(const_trait_impl)]
//! # use const_soft_float::soft_f32::SoftF32;
//! const fn const_f32_add(a: f32, b: f32) -> f32 {
//!     (SoftF32(a) + SoftF32(b)).to_f32()
//! }
//! # }
//! # }
//! ```
//!
//! with `const_mut_refs` usage (requires `nightly`):
//! ```
//! # cfg_if::cfg_if! {
//! # if #[cfg(nightly)] {
//! # #![feature(const_trait_impl)]
//! # #![feature(const_mut_refs)]
//! # use const_soft_float::soft_f32::SoftF32;
//! const fn const_f32_add(a: f32, b: f32) -> f32 {
//!     let mut x = SoftF32(a);
//!     x += SoftF32(b);
//!     x.to_f32()
//! }
//! # }
//! # }
//! ```
//!
//!

#![cfg_attr(feature = "no_std", no_std)]
#![cfg_attr(feature = "const_trait_impl", feature(const_trait_impl))]
#![cfg_attr(feature = "const_mut_refs", feature(const_mut_refs))]

pub mod soft_f32;
pub mod soft_f64;

const fn abs_diff(a: i32, b: i32) -> u32 {
    a.wrapping_sub(b).wrapping_abs() as u32
}

#[cfg(test)]
mod tests {
    use crate::soft_f32::SoftF32;
    use crate::soft_f64::SoftF64;

    const RANGE: core::ops::Range<i32> = -1000..1000;
    const F32_FACTOR: f32 = 10.0;
    const F64_FACTOR: f64 = 1000.0;

    #[test]
    fn f32_add() {
        for a in RANGE {
            let a = a as f32 * F32_FACTOR;
            for b in RANGE {
                let b = b as f32 * F32_FACTOR;
                assert_eq!(SoftF32(a).add(SoftF32(b)).0, a + b);
            }
        }
    }

    #[test]
    fn f32_sub() {
        for a in RANGE {
            let a = a as f32 * F32_FACTOR;
            for b in RANGE {
                let b = b as f32 * F32_FACTOR;
                assert_eq!(SoftF32(a).sub(SoftF32(b)).0, a - b);
            }
        }
    }

    #[test]
    fn f32_mul() {
        for a in RANGE {
            let a = a as f32 * F32_FACTOR;
            for b in RANGE {
                let b = b as f32 * F32_FACTOR;
                assert_eq!(SoftF32(a).mul(SoftF32(b)).0, a * b);
            }
        }
    }

    #[test]
    fn f32_div() {
        for a in RANGE {
            let a = a as f32 * F32_FACTOR;
            for b in RANGE {
                let b = b as f32 * F32_FACTOR;
                let x = SoftF32(a).div(SoftF32(b)).0;
                let y = a / b;
                assert!(x == y || x.is_nan() && y.is_nan())
            }
        }
    }

    #[test]
    fn f64_add() {
        for a in RANGE {
            let a = a as f64 * F64_FACTOR;
            for b in RANGE {
                let b = b as f64 * F64_FACTOR;
                assert_eq!(SoftF64(a).sub(SoftF64(b)).0, a - b);
            }
        }
    }

    #[test]
    fn f64_sub() {
        for a in RANGE {
            let a = a as f64 * F64_FACTOR;
            for b in RANGE {
                let b = b as f64 * F64_FACTOR;
                assert_eq!(SoftF64(a).sub(SoftF64(b)).0, a - b);
            }
        }
    }

    #[test]
    fn f64_mul() {
        for a in RANGE {
            let a = a as f64 * F64_FACTOR;
            for b in RANGE {
                let b = b as f64 * F64_FACTOR;
                assert_eq!(SoftF64(a).mul(SoftF64(b)).0, a * b);
            }
        }
    }

    #[test]
    fn f64_div() {
        for a in RANGE {
            let a = a as f64 * F64_FACTOR;
            for b in RANGE {
                let b = b as f64 * F64_FACTOR;
                let x = SoftF64(a).div(SoftF64(b)).0;
                let y = a / b;
                assert!(x == y || x.is_nan() && y.is_nan())
            }
        }
    }
}
