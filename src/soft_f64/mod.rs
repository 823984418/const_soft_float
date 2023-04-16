pub(crate) mod helpers;

pub mod add;
pub mod cmp;
pub mod copysign;
pub mod cos;
pub mod div;
pub mod floor;
pub mod mul;
pub mod pow;
pub mod round;
pub mod sin;
pub mod sqrt;
pub mod trunc;

#[cfg(feature = "const_trait_impl")]
pub mod const_impl_trait;

#[cfg(feature = "const_trait_impl")]
pub use const_impl_trait as impl_trait;

#[cfg(not(feature = "const_trait_impl"))]
pub mod impl_trait;

#[derive(Default, Copy, Clone)]
#[repr(transparent)]
pub struct SoftF64(pub f64);

impl SoftF64 {
    pub const fn from_f64(a: f64) -> Self {
        Self(a)
    }

    pub const fn to_f64(self) -> f64 {
        self.0
    }

    pub const fn from_bits(a: u64) -> Self {
        Self(unsafe { core::mem::transmute(a) })
    }

    pub const fn to_bits(self) -> u64 {
        unsafe { core::mem::transmute(self.0) }
    }

    pub const fn add(self, rhs: Self) -> Self {
        add::add(self, rhs)
    }

    pub const fn mul(self, rhs: Self) -> Self {
        mul::mul(self, rhs)
    }

    pub const fn div(self, rhs: Self) -> Self {
        div::div(self, rhs)
    }

    pub const fn cmp(self, rhs: Self) -> Option<core::cmp::Ordering> {
        cmp::cmp(self, rhs)
    }

    pub const fn neg(self) -> Self {
        Self::from_repr(self.repr() ^ Self::SIGN_MASK)
    }

    pub const fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    pub const fn sqrt(self) -> Self {
        sqrt::sqrt(self)
    }

    pub const fn powi(self, n: i32) -> Self {
        pow::pow(self, n)
    }

    pub const fn copysign(self, other: Self) -> Self {
        copysign::copysign(self, other)
    }

    pub const fn trunc(self) -> Self {
        trunc::trunc(self)
    }

    pub const fn round(self) -> Self {
        round::round(self)
    }

    pub const fn floor(self) -> Self {
        floor::floor(self)
    }

    pub const fn sin(self) -> Self {
        sin::sin(self)
    }

    pub const fn cos(self) -> Self {
        cos::cos(self)
    }
}

type SelfInt = u64;
type SelfSignedInt = i64;
type SelfExpInt = i16;

#[allow(unused)]
impl SoftF64 {
    const ZERO: Self = Self(0.0);
    const ONE: Self = Self(1.0);
    const BITS: u32 = 64;
    const SIGNIFICAND_BITS: u32 = 52;
    const EXPONENT_BITS: u32 = Self::BITS - Self::SIGNIFICAND_BITS - 1;
    const EXPONENT_MAX: u32 = (1 << Self::EXPONENT_BITS) - 1;
    const EXPONENT_BIAS: u32 = Self::EXPONENT_MAX >> 1;
    const SIGN_MASK: SelfInt = 1 << (Self::BITS - 1);
    const SIGNIFICAND_MASK: SelfInt = (1 << Self::SIGNIFICAND_BITS) - 1;
    const IMPLICIT_BIT: SelfInt = 1 << Self::SIGNIFICAND_BITS;
    const EXPONENT_MASK: SelfInt = !(Self::SIGN_MASK | Self::SIGNIFICAND_MASK);

    const fn repr(self) -> SelfInt {
        self.to_bits()
    }
    const fn signed_repr(self) -> SelfSignedInt {
        self.to_bits() as SelfSignedInt
    }
    const fn sign(self) -> bool {
        self.signed_repr() < 0
    }
    const fn exp(self) -> SelfExpInt {
        ((self.to_bits() & Self::EXPONENT_MASK) >> Self::SIGNIFICAND_BITS) as SelfExpInt
    }
    const fn frac(self) -> SelfInt {
        self.to_bits() & Self::SIGNIFICAND_MASK
    }
    const fn imp_frac(self) -> SelfInt {
        self.frac() | Self::IMPLICIT_BIT
    }
    const fn from_repr(a: SelfInt) -> Self {
        Self::from_bits(a)
    }
    const fn from_parts(sign: bool, exponent: SelfInt, significand: SelfInt) -> Self {
        Self::from_repr(
            ((sign as SelfInt) << (Self::BITS - 1))
                | ((exponent << Self::SIGNIFICAND_BITS) & Self::EXPONENT_MASK)
                | (significand & Self::SIGNIFICAND_MASK),
        )
    }
    const fn normalize(significand: SelfInt) -> (i32, SelfInt) {
        let shift = significand
            .leading_zeros()
            .wrapping_sub((1u64 << Self::SIGNIFICAND_BITS).leading_zeros());
        (
            1i32.wrapping_sub(shift as i32),
            significand << shift as SelfInt,
        )
    }
    const fn is_subnormal(self) -> bool {
        (self.repr() & Self::EXPONENT_MASK) == 0
    }

    const fn scalbn(self, n: i32) -> SoftF64 {
        helpers::scalbn(self, n)
    }
}

const fn u128_lo(x: u128) -> u64 {
    x as u64
}

const fn u128_hi(x: u128) -> u64 {
    (x >> 64) as u64
}

const fn u64_widen_mul(a: u64, b: u64) -> (u64, u64) {
    let x = u128::wrapping_mul(a as _, b as _);
    (u128_lo(x), u128_hi(x))
}
