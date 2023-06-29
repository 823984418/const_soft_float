#![cfg(test)]
// A local copy of the compiler_builtins source code

use core::ops;
pub(crate) trait Float:
    Copy
    + core::fmt::Debug
    + PartialEq
    + PartialOrd
    + ops::AddAssign
    + ops::MulAssign
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Div<Output = Self>
    + ops::Rem<Output = Self>
{
    /// A uint of the same width as the float
    type Int: Int;

    /// A int of the same width as the float
    type SignedInt: Int;

    /// An int capable of containing the exponent bits plus a sign bit. This is signed.
    type ExpInt: Int;

    const ZERO: Self;
    const ONE: Self;

    /// The bitwidth of the float type
    const BITS: u32;

    /// The bitwidth of the significand
    const SIGNIFICAND_BITS: u32;

    /// The bitwidth of the exponent
    const EXPONENT_BITS: u32 = Self::BITS - Self::SIGNIFICAND_BITS - 1;

    /// The maximum value of the exponent
    const EXPONENT_MAX: u32 = (1 << Self::EXPONENT_BITS) - 1;

    /// The exponent bias value
    const EXPONENT_BIAS: u32 = Self::EXPONENT_MAX >> 1;

    /// A mask for the sign bit
    const SIGN_MASK: Self::Int;

    /// A mask for the significand
    const SIGNIFICAND_MASK: Self::Int;

    // The implicit bit of the float format
    const IMPLICIT_BIT: Self::Int;

    /// A mask for the exponent
    const EXPONENT_MASK: Self::Int;

    /// Returns `self` transmuted to `Self::Int`
    fn repr(self) -> Self::Int;

    /// Returns `self` transmuted to `Self::SignedInt`
    fn signed_repr(self) -> Self::SignedInt;

    /// Checks if two floats have the same bit representation. *Except* for NaNs! NaN can be
    /// represented in multiple different ways. This method returns `true` if two NaNs are
    /// compared.
    fn eq_repr(self, rhs: Self) -> bool;

    /// Returns the sign bit
    fn sign(self) -> bool;

    /// Returns the exponent with bias
    fn exp(self) -> Self::ExpInt;

    /// Returns the significand with no implicit bit (or the "fractional" part)
    fn frac(self) -> Self::Int;

    /// Returns the significand with implicit bit
    fn imp_frac(self) -> Self::Int;

    /// Returns a `Self::Int` transmuted back to `Self`
    fn from_repr(a: Self::Int) -> Self;

    /// Constructs a `Self` from its parts. Inputs are treated as bits and shifted into position.
    fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self;

    /// Returns (normalized exponent, normalized significand)
    fn normalize(significand: Self::Int) -> (i32, Self::Int);

    /// Returns if `self` is subnormal
    fn is_subnormal(self) -> bool;
}

macro_rules! float_impl {
    ($ty:ident, $ity:ident, $sity:ident, $expty:ident, $bits:expr, $significand_bits:expr) => {
        impl Float for $ty {
            type Int = $ity;
            type SignedInt = $sity;
            type ExpInt = $expty;

            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            const BITS: u32 = $bits;
            const SIGNIFICAND_BITS: u32 = $significand_bits;

            const SIGN_MASK: Self::Int = 1 << (Self::BITS - 1);
            const SIGNIFICAND_MASK: Self::Int = (1 << Self::SIGNIFICAND_BITS) - 1;
            const IMPLICIT_BIT: Self::Int = 1 << Self::SIGNIFICAND_BITS;
            const EXPONENT_MASK: Self::Int = !(Self::SIGN_MASK | Self::SIGNIFICAND_MASK);

            fn repr(self) -> Self::Int {
                self.to_bits()
            }
            fn signed_repr(self) -> Self::SignedInt {
                self.to_bits() as Self::SignedInt
            }
            fn eq_repr(self, rhs: Self) -> bool {
                if self.is_nan() && rhs.is_nan() {
                    true
                } else {
                    self.repr() == rhs.repr()
                }
            }
            fn sign(self) -> bool {
                self.signed_repr() < Self::SignedInt::ZERO
            }
            fn exp(self) -> Self::ExpInt {
                ((self.to_bits() & Self::EXPONENT_MASK) >> Self::SIGNIFICAND_BITS) as Self::ExpInt
            }
            fn frac(self) -> Self::Int {
                self.to_bits() & Self::SIGNIFICAND_MASK
            }
            fn imp_frac(self) -> Self::Int {
                self.frac() | Self::IMPLICIT_BIT
            }
            fn from_repr(a: Self::Int) -> Self {
                Self::from_bits(a)
            }
            fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self {
                Self::from_repr(
                    ((sign as Self::Int) << (Self::BITS - 1))
                        | ((exponent << Self::SIGNIFICAND_BITS) & Self::EXPONENT_MASK)
                        | (significand & Self::SIGNIFICAND_MASK),
                )
            }
            fn normalize(significand: Self::Int) -> (i32, Self::Int) {
                let shift = significand
                    .leading_zeros()
                    .wrapping_sub((Self::Int::ONE << Self::SIGNIFICAND_BITS).leading_zeros());
                (
                    1i32.wrapping_sub(shift as i32),
                    significand << shift as Self::Int,
                )
            }
            fn is_subnormal(self) -> bool {
                (self.repr() & Self::EXPONENT_MASK) == Self::Int::ZERO
            }
        }
    };
}

float_impl!(f32, u32, i32, i16, 32, 23);
float_impl!(f64, u64, i64, i16, 64, 52);

pub(crate) trait Int:
    Copy
    + core::fmt::Debug
    + PartialEq
    + PartialOrd
    + ops::AddAssign
    + ops::SubAssign
    + ops::BitAndAssign
    + ops::BitOrAssign
    + ops::BitXorAssign
    + ops::ShlAssign<i32>
    + ops::ShrAssign<u32>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Div<Output = Self>
    + ops::Shl<u32, Output = Self>
    + ops::Shr<u32, Output = Self>
    + ops::BitOr<Output = Self>
    + ops::BitXor<Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::Not<Output = Self>
{
    /// Type with the same width but other signedness
    type OtherSign: Int;
    /// Unsigned version of Self
    type UnsignedInt: Int;

    /// If `Self` is a signed integer
    const SIGNED: bool;

    /// The bitwidth of the int type
    const BITS: u32;

    const ZERO: Self;
    const ONE: Self;
    const MIN: Self;
    const MAX: Self;

    /// LUT used for maximizing the space covered and minimizing the computational cost of fuzzing
    /// in `testcrate`. For example, Self = u128 produces [0,1,2,7,8,15,16,31,32,63,64,95,96,111,
    /// 112,119,120,125,126,127].
    const FUZZ_LENGTHS: [u8; 20];
    /// The number of entries of `FUZZ_LENGTHS` actually used. The maximum is 20 for u128.
    const FUZZ_NUM: usize;

    fn unsigned(self) -> Self::UnsignedInt;
    fn from_unsigned(unsigned: Self::UnsignedInt) -> Self;

    fn from_bool(b: bool) -> Self;

    /// Prevents the need for excessive conversions between signed and unsigned
    fn logical_shr(self, other: u32) -> Self;

    /// Absolute difference between two integers.
    fn abs_diff(self, other: Self) -> Self::UnsignedInt;

    // copied from primitive integers, but put in a trait
    fn is_zero(self) -> bool;
    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_shl(self, other: u32) -> Self;
    fn wrapping_shr(self, other: u32) -> Self;
    fn rotate_left(self, other: u32) -> Self;
    fn overflowing_add(self, other: Self) -> (Self, bool);
    fn leading_zeros(self) -> u32;
}

macro_rules! int_impl_common {
    ($ty:ty) => {
        const BITS: u32 = <Self as Int>::ZERO.count_zeros();
        const SIGNED: bool = Self::MIN != Self::ZERO;

        const ZERO: Self = 0;
        const ONE: Self = 1;
        const MIN: Self = <Self>::MIN;
        const MAX: Self = <Self>::MAX;

        const FUZZ_LENGTHS: [u8; 20] = {
            let bits = <Self as Int>::BITS;
            let mut v = [0u8; 20];
            v[0] = 0;
            v[1] = 1;
            v[2] = 2; // important for parity and the iX::MIN case when reversed
            let mut i = 3;
            // No need for any more until the byte boundary, because there should be no algorithms
            // that are sensitive to anything not next to byte boundaries after 2. We also scale
            // in powers of two, which is important to prevent u128 corner tests from getting too
            // big.
            let mut l = 8;
            loop {
                if l >= ((bits / 2) as u8) {
                    break;
                }
                // get both sides of the byte boundary
                v[i] = l - 1;
                i += 1;
                v[i] = l;
                i += 1;
                l *= 2;
            }

            if bits != 8 {
                // add the lower side of the middle boundary
                v[i] = ((bits / 2) - 1) as u8;
                i += 1;
            }

            // We do not want to jump directly from the Self::BITS/2 boundary to the Self::BITS
            // boundary because of algorithms that split the high part up. We reverse the scaling
            // as we go to Self::BITS.
            let mid = i;
            let mut j = 1;
            loop {
                v[i] = (bits as u8) - (v[mid - j]) - 1;
                if j == mid {
                    break;
                }
                i += 1;
                j += 1;
            }
            v
        };

        const FUZZ_NUM: usize = {
            let log2 = (<Self as Int>::BITS - 1).count_ones() as usize;
            if log2 == 3 {
                // case for u8
                6
            } else {
                // 3 entries on each extreme, 2 in the middle, and 4 for each scale of intermediate
                // boundaries.
                8 + (4 * (log2 - 4))
            }
        };

        fn from_bool(b: bool) -> Self {
            b as $ty
        }

        fn logical_shr(self, other: u32) -> Self {
            Self::from_unsigned(self.unsigned().wrapping_shr(other))
        }

        fn is_zero(self) -> bool {
            self == Self::ZERO
        }

        fn wrapping_neg(self) -> Self {
            <Self>::wrapping_neg(self)
        }

        fn wrapping_add(self, other: Self) -> Self {
            <Self>::wrapping_add(self, other)
        }

        fn wrapping_mul(self, other: Self) -> Self {
            <Self>::wrapping_mul(self, other)
        }

        fn wrapping_sub(self, other: Self) -> Self {
            <Self>::wrapping_sub(self, other)
        }

        fn wrapping_shl(self, other: u32) -> Self {
            <Self>::wrapping_shl(self, other)
        }

        fn wrapping_shr(self, other: u32) -> Self {
            <Self>::wrapping_shr(self, other)
        }

        fn rotate_left(self, other: u32) -> Self {
            <Self>::rotate_left(self, other)
        }

        fn overflowing_add(self, other: Self) -> (Self, bool) {
            <Self>::overflowing_add(self, other)
        }

        fn leading_zeros(self) -> u32 {
            <Self>::leading_zeros(self)
        }
    };
}

macro_rules! int_impl {
    ($ity:ty, $uty:ty) => {
        impl Int for $uty {
            type OtherSign = $ity;
            type UnsignedInt = $uty;

            fn unsigned(self) -> $uty {
                self
            }

            // It makes writing macros easier if this is implemented for both signed and unsigned
            #[allow(clippy::wrong_self_convention)]
            fn from_unsigned(me: $uty) -> Self {
                me
            }

            fn abs_diff(self, other: Self) -> Self {
                if self < other {
                    other.wrapping_sub(self)
                } else {
                    self.wrapping_sub(other)
                }
            }

            int_impl_common!($uty);
        }

        impl Int for $ity {
            type OtherSign = $uty;
            type UnsignedInt = $uty;

            fn unsigned(self) -> $uty {
                self as $uty
            }

            fn from_unsigned(me: $uty) -> Self {
                me as $ity
            }

            fn abs_diff(self, other: Self) -> $uty {
                self.wrapping_sub(other).wrapping_abs() as $uty
            }

            int_impl_common!($ity);
        }
    };
}

int_impl!(isize, usize);
int_impl!(i8, u8);
int_impl!(i16, u16);
int_impl!(i32, u32);
int_impl!(i64, u64);
int_impl!(i128, u128);

pub(crate) trait CastInto<T: Copy>: Copy {
    fn cast(self) -> T;
}

macro_rules! cast_into {
    ($ty:ty) => {
        cast_into!($ty; usize, isize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);
    };
    ($ty:ty; $($into:ty),*) => {$(
        impl CastInto<$into> for $ty {
            fn cast(self) -> $into {
                self as $into
            }
        }
    )*};
}

cast_into!(usize);
cast_into!(isize);
cast_into!(u8);
cast_into!(i8);
cast_into!(u16);
cast_into!(i16);
cast_into!(u32);
cast_into!(i32);
cast_into!(u64);
cast_into!(i64);
cast_into!(u128);
cast_into!(i128);

pub(crate) trait HInt: Int {
    /// Integer that is double the bit width of the integer this trait is implemented for
    type D: DInt<H = Self> + Int;

    /// Widens (using default extension) the integer to have double bit width
    fn widen(self) -> Self::D;
    /// Widens (zero extension only) the integer to have double bit width. This is needed to get
    /// around problems with associated type bounds (such as `Int<Othersign: DInt>`) being unstable
    fn zero_widen(self) -> Self::D;
    /// Widens the integer to have double bit width and shifts the integer into the higher bits
    fn widen_hi(self) -> Self::D;
    /// Widening multiplication with zero widening. This cannot overflow.
    fn zero_widen_mul(self, rhs: Self) -> Self::D;
    /// Widening multiplication. This cannot overflow.
    fn widen_mul(self, rhs: Self) -> Self::D;
}

/// Trait for integers twice the bit width of another integer. This is implemented for all
/// primitives except for `u8`, because there is not a smaller primitive.
pub(crate) trait DInt: Int {
    /// Integer that is half the bit width of the integer this trait is implemented for
    type H: HInt<D = Self> + Int;

    /// Returns the low half of `self`
    fn lo(self) -> Self::H;
    /// Returns the high half of `self`
    fn hi(self) -> Self::H;
    /// Returns the low and high halves of `self` as a tuple
    fn lo_hi(self) -> (Self::H, Self::H);
    /// Constructs an integer using lower and higher half parts
    fn from_lo_hi(lo: Self::H, hi: Self::H) -> Self;
}

macro_rules! impl_h_int {
    ($($H:ident $uH:ident $X:ident),*) => {
        $(
            impl HInt for $H {
                type D = $X;

                fn widen(self) -> Self::D {
                    self as $X
                }
                fn zero_widen(self) -> Self::D {
                    (self as $uH) as $X
                }
                fn widen_hi(self) -> Self::D {
                    (self as $X) << <$H as Int>::BITS
                }
                fn zero_widen_mul(self, rhs: Self) -> Self::D {
                    self.zero_widen().wrapping_mul(rhs.zero_widen())
                }
                fn widen_mul(self, rhs: Self) -> Self::D {
                    self.widen().wrapping_mul(rhs.widen())
                }
            }
        )*
    };
}

macro_rules! impl_d_int {
    ($($X:ident $D:ident),*) => {
        $(
            impl DInt for $D {
                type H = $X;

                fn lo(self) -> Self::H {
                    self as $X
                }
                fn hi(self) -> Self::H {
                    (self >> <$X as Int>::BITS) as $X
                }
                fn lo_hi(self) -> (Self::H, Self::H) {
                    (self.lo(), self.hi())
                }
                fn from_lo_hi(lo: Self::H, hi: Self::H) -> Self {
                    lo.zero_widen() | hi.widen_hi()
                }
            }
        )*
    };
}

impl_d_int!(u8 u16, u16 u32, u32 u64, u64 u128, i8 i16, i16 i32, i32 i64, i64 i128);
impl_h_int!(
    u8 u8 u16,
    u16 u16 u32,
    u32 u32 u64,
    u64 u64 u128,
    i8 u8 i16,
    i16 u16 i32,
    i32 u32 i64,
    i64 u64 i128
);

/// Returns `a + b`
pub(crate) fn add<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
{
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;

    let bits = F::BITS.cast();
    let significand_bits = F::SIGNIFICAND_BITS;
    let max_exponent = F::EXPONENT_MAX;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIGNIFICAND_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;

    let mut a_rep = a.repr();
    let mut b_rep = b.repr();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    // Detect if a or b is zero, infinity, or NaN.
    if a_abs.wrapping_sub(one) >= inf_rep - one || b_abs.wrapping_sub(one) >= inf_rep - one {
        // NaN + anything = qNaN
        if a_abs > inf_rep {
            return F::from_repr(a_abs | quiet_bit);
        }
        // anything + NaN = qNaN
        if b_abs > inf_rep {
            return F::from_repr(b_abs | quiet_bit);
        }

        if a_abs == inf_rep {
            // +/-infinity + -/+infinity = qNaN
            if (a.repr() ^ b.repr()) == sign_bit {
                return F::from_repr(qnan_rep);
            } else {
                // +/-infinity + anything remaining = +/- infinity
                return a;
            }
        }

        // anything remaining + +/-infinity = +/-infinity
        if b_abs == inf_rep {
            return b;
        }

        // zero + anything = anything
        if a_abs == Int::ZERO {
            // but we need to get the sign right for zero + zero
            if b_abs == Int::ZERO {
                return F::from_repr(a.repr() & b.repr());
            } else {
                return b;
            }
        }

        // anything + zero = anything
        if b_abs == Int::ZERO {
            return a;
        }
    }

    // Swap a and b if necessary so that a has the larger absolute value.
    if b_abs > a_abs {
        // Don't use mem::swap because it may generate references to memcpy in unoptimized code.
        let tmp = a_rep;
        a_rep = b_rep;
        b_rep = tmp;
    }

    // Extract the exponent and significand from the (possibly swapped) a and b.
    let mut a_exponent: i32 = ((a_rep & exponent_mask) >> significand_bits).cast();
    let mut b_exponent: i32 = ((b_rep & exponent_mask) >> significand_bits).cast();
    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;

    // normalize any denormals, and adjust the exponent accordingly.
    if a_exponent == 0 {
        let (exponent, significand) = F::normalize(a_significand);
        a_exponent = exponent;
        a_significand = significand;
    }
    if b_exponent == 0 {
        let (exponent, significand) = F::normalize(b_significand);
        b_exponent = exponent;
        b_significand = significand;
    }

    // The sign of the result is the sign of the larger operand, a.  If they
    // have opposite signs, we are performing a subtraction; otherwise addition.
    let result_sign = a_rep & sign_bit;
    let subtraction = ((a_rep ^ b_rep) & sign_bit) != zero;

    // Shift the significands to give us round, guard and sticky, and or in the
    // implicit significand bit.  (If we fell through from the denormal path it
    // was already set by normalize(), but setting it twice won't hurt
    // anything.)
    a_significand = (a_significand | implicit_bit) << 3;
    b_significand = (b_significand | implicit_bit) << 3;

    // Shift the significand of b by the difference in exponents, with a sticky
    // bottom bit to get rounding correct.
    let align = a_exponent.wrapping_sub(b_exponent).cast();
    if align != Int::ZERO {
        if align < bits {
            let sticky =
                F::Int::from_bool(b_significand << bits.wrapping_sub(align).cast() != Int::ZERO);
            b_significand = (b_significand >> align.cast()) | sticky;
        } else {
            b_significand = one; // sticky; b is known to be non-zero.
        }
    }
    if subtraction {
        a_significand = a_significand.wrapping_sub(b_significand);
        // If a == -b, return +zero.
        if a_significand == Int::ZERO {
            return F::from_repr(Int::ZERO);
        }

        // If partial cancellation occured, we need to left-shift the result
        // and adjust the exponent:
        if a_significand < implicit_bit << 3 {
            let shift =
                a_significand.leading_zeros() as i32 - (implicit_bit << 3).leading_zeros() as i32;
            a_significand <<= shift;
            a_exponent -= shift;
        }
    } else {
        // addition
        a_significand += b_significand;

        // If the addition carried up, we need to right-shift the result and
        // adjust the exponent:
        if a_significand & implicit_bit << 4 != Int::ZERO {
            let sticky = F::Int::from_bool(a_significand & one != Int::ZERO);
            a_significand = a_significand >> 1 | sticky;
            a_exponent += 1;
        }
    }

    // If we have overflowed the type, return +/- infinity:
    if a_exponent >= max_exponent as i32 {
        return F::from_repr(inf_rep | result_sign);
    }

    if a_exponent <= 0 {
        // Result is denormal before rounding; the exponent is zero and we
        // need to shift the significand.
        let shift = (1 - a_exponent).cast();
        let sticky =
            F::Int::from_bool((a_significand << bits.wrapping_sub(shift).cast()) != Int::ZERO);
        a_significand = a_significand >> shift.cast() | sticky;
        a_exponent = 0;
    }

    // Low three bits are round, guard, and sticky.
    let a_significand_i32: i32 = a_significand.cast();
    let round_guard_sticky: i32 = a_significand_i32 & 0x7;

    // Shift the significand into place, and mask off the implicit bit.
    let mut result = a_significand >> 3 & significand_mask;

    // Insert the exponent and sign.
    result |= a_exponent.cast() << significand_bits;
    result |= result_sign;

    // Final rounding.  The result may overflow to infinity, but that is the
    // correct result in that case.
    if round_guard_sticky > 0x4 {
        result += one;
    }
    if round_guard_sticky == 0x4 {
        result += result & one;
    }

    F::from_repr(result)
}

pub(crate) fn div32<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
    F::Int: HInt,
    <F as Float>::Int: core::ops::Mul,
{
    const NUMBER_OF_HALF_ITERATIONS: usize = 0;
    const NUMBER_OF_FULL_ITERATIONS: usize = 3;
    const USE_NATIVE_FULL_ITERATIONS: bool = true;

    let one = F::Int::ONE;
    let zero = F::Int::ZERO;
    let hw = F::BITS / 2;
    let lo_mask = u32::MAX >> hw;

    let significand_bits = F::SIGNIFICAND_BITS;
    let max_exponent = F::EXPONENT_MAX;

    let exponent_bias = F::EXPONENT_BIAS;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIGNIFICAND_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;

    #[inline(always)]
    fn negate_u32(a: u32) -> u32 {
        (<i32>::wrapping_neg(a as i32)) as u32
    }

    let a_rep = a.repr();
    let b_rep = b.repr();

    let a_exponent = (a_rep >> significand_bits) & max_exponent.cast();
    let b_exponent = (b_rep >> significand_bits) & max_exponent.cast();
    let quotient_sign = (a_rep ^ b_rep) & sign_bit;

    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;
    let mut scale = 0;

    // Detect if a or b is zero, denormal, infinity, or NaN.
    if a_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
        || b_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
    {
        let a_abs = a_rep & abs_mask;
        let b_abs = b_rep & abs_mask;

        // NaN / anything = qNaN
        if a_abs > inf_rep {
            return F::from_repr(a_rep | quiet_bit);
        }
        // anything / NaN = qNaN
        if b_abs > inf_rep {
            return F::from_repr(b_rep | quiet_bit);
        }

        if a_abs == inf_rep {
            if b_abs == inf_rep {
                // infinity / infinity = NaN
                return F::from_repr(qnan_rep);
            } else {
                // infinity / anything else = +/- infinity
                return F::from_repr(a_abs | quotient_sign);
            }
        }

        // anything else / infinity = +/- 0
        if b_abs == inf_rep {
            return F::from_repr(quotient_sign);
        }

        if a_abs == zero {
            if b_abs == zero {
                // zero / zero = NaN
                return F::from_repr(qnan_rep);
            } else {
                // zero / anything else = +/- zero
                return F::from_repr(quotient_sign);
            }
        }

        // anything else / zero = +/- infinity
        if b_abs == zero {
            return F::from_repr(inf_rep | quotient_sign);
        }

        // one or both of a or b is denormal, the other (if applicable) is a
        // normal number.  Renormalize one or both of a and b, and set scale to
        // include the necessary exponent adjustment.
        if a_abs < implicit_bit {
            let (exponent, significand) = F::normalize(a_significand);
            scale += exponent;
            a_significand = significand;
        }

        if b_abs < implicit_bit {
            let (exponent, significand) = F::normalize(b_significand);
            scale -= exponent;
            b_significand = significand;
        }
    }

    // Set the implicit significand bit.  If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;

    let written_exponent: i32 = CastInto::<u32>::cast(
        a_exponent
            .wrapping_sub(b_exponent)
            .wrapping_add(scale.cast()),
    )
    .wrapping_add(exponent_bias) as i32;
    let b_uq1 = b_significand << (F::BITS - significand_bits - 1);

    // Align the significand of b as a UQ1.(n-1) fixed-point number in the range
    // [1.0, 2.0) and get a UQ0.n approximate reciprocal using a small minimax
    // polynomial approximation: x0 = 3/4 + 1/sqrt(2) - b/2.
    // The max error for this approximation is achieved at endpoints, so
    //   abs(x0(b) - 1/b) <= abs(x0(1) - 1/1) = 3/4 - 1/sqrt(2) = 0.04289...,
    // which is about 4.5 bits.
    // The initial approximation is between x0(1.0) = 0.9571... and x0(2.0) = 0.4571...

    // Then, refine the reciprocal estimate using a quadratically converging
    // Newton-Raphson iteration:
    //     x_{n+1} = x_n * (2 - x_n * b)
    //
    // Let b be the original divisor considered "in infinite precision" and
    // obtained from IEEE754 representation of function argument (with the
    // implicit bit set). Corresponds to rep_t-sized b_UQ1 represented in
    // UQ1.(W-1).
    //
    // Let b_hw be an infinitely precise number obtained from the highest (HW-1)
    // bits of divisor significand (with the implicit bit set). Corresponds to
    // half_rep_t-sized b_UQ1_hw represented in UQ1.(HW-1) that is a **truncated**
    // version of b_UQ1.
    //
    // Let e_n := x_n - 1/b_hw
    //     E_n := x_n - 1/b
    // abs(E_n) <= abs(e_n) + (1/b_hw - 1/b)
    //           = abs(e_n) + (b - b_hw) / (b*b_hw)
    //          <= abs(e_n) + 2 * 2^-HW

    // rep_t-sized iterations may be slower than the corresponding half-width
    // variant depending on the handware and whether single/double/quad precision
    // is selected.
    // NB: Using half-width iterations increases computation errors due to
    // rounding, so error estimations have to be computed taking the selected
    // mode into account!

    #[allow(clippy::absurd_extreme_comparisons)]
    let mut x_uq0 = if NUMBER_OF_HALF_ITERATIONS > 0 {
        // Starting with (n-1) half-width iterations
        let b_uq1_hw: u16 =
            (CastInto::<u32>::cast(b_significand) >> (significand_bits + 1 - hw)) as u16;

        // C is (3/4 + 1/sqrt(2)) - 1 truncated to W0 fractional bits as UQ0.HW
        // with W0 being either 16 or 32 and W0 <= HW.
        // That is, C is the aforementioned 3/4 + 1/sqrt(2) constant (from which
        // b/2 is subtracted to obtain x0) wrapped to [0, 1) range.

        // HW is at least 32. Shifting into the highest bits if needed.
        let c_hw = (0x7504_u32 as u16).wrapping_shl(hw.wrapping_sub(32));

        // b >= 1, thus an upper bound for 3/4 + 1/sqrt(2) - b/2 is about 0.9572,
        // so x0 fits to UQ0.HW without wrapping.
        let x_uq0_hw: u16 = {
            let mut x_uq0_hw: u16 = c_hw.wrapping_sub(b_uq1_hw /* exact b_hw/2 as UQ0.HW */);
            // An e_0 error is comprised of errors due to
            // * x0 being an inherently imprecise first approximation of 1/b_hw
            // * C_hw being some (irrational) number **truncated** to W0 bits
            // Please note that e_0 is calculated against the infinitely precise
            // reciprocal of b_hw (that is, **truncated** version of b).
            //
            // e_0 <= 3/4 - 1/sqrt(2) + 2^-W0

            // By construction, 1 <= b < 2
            // f(x)  = x * (2 - b*x) = 2*x - b*x^2
            // f'(x) = 2 * (1 - b*x)
            //
            // On the [0, 1] interval, f(0)   = 0,
            // then it increses until  f(1/b) = 1 / b, maximum on (0, 1),
            // then it decreses to     f(1)   = 2 - b
            //
            // Let g(x) = x - f(x) = b*x^2 - x.
            // On (0, 1/b), g(x) < 0 <=> f(x) > x
            // On (1/b, 1], g(x) > 0 <=> f(x) < x
            //
            // For half-width iterations, b_hw is used instead of b.
            #[allow(clippy::reversed_empty_ranges)]
            for _ in 0..NUMBER_OF_HALF_ITERATIONS {
                // corr_UQ1_hw can be **larger** than 2 - b_hw*x by at most 1*Ulp
                // of corr_UQ1_hw.
                // "0.0 - (...)" is equivalent to "2.0 - (...)" in UQ1.(HW-1).
                // On the other hand, corr_UQ1_hw should not overflow from 2.0 to 0.0 provided
                // no overflow occurred earlier: ((rep_t)x_UQ0_hw * b_UQ1_hw >> HW) is
                // expected to be strictly positive because b_UQ1_hw has its highest bit set
                // and x_UQ0_hw should be rather large (it converges to 1/2 < 1/b_hw <= 1).
                let corr_uq1_hw: u16 =
                    0.wrapping_sub((x_uq0_hw as u32).wrapping_mul(b_uq1_hw.cast()) >> hw) as u16;

                // Now, we should multiply UQ0.HW and UQ1.(HW-1) numbers, naturally
                // obtaining an UQ1.(HW-1) number and proving its highest bit could be
                // considered to be 0 to be able to represent it in UQ0.HW.
                // From the above analysis of f(x), if corr_UQ1_hw would be represented
                // without any intermediate loss of precision (that is, in twice_rep_t)
                // x_UQ0_hw could be at most [1.]000... if b_hw is exactly 1.0 and strictly
                // less otherwise. On the other hand, to obtain [1.]000..., one have to pass
                // 1/b_hw == 1.0 to f(x), so this cannot occur at all without overflow (due
                // to 1.0 being not representable as UQ0.HW).
                // The fact corr_UQ1_hw was virtually round up (due to result of
                // multiplication being **first** truncated, then negated - to improve
                // error estimations) can increase x_UQ0_hw by up to 2*Ulp of x_UQ0_hw.
                x_uq0_hw = ((x_uq0_hw as u32).wrapping_mul(corr_uq1_hw as u32) >> (hw - 1)) as u16;
                // Now, either no overflow occurred or x_UQ0_hw is 0 or 1 in its half_rep_t
                // representation. In the latter case, x_UQ0_hw will be either 0 or 1 after
                // any number of iterations, so just subtract 2 from the reciprocal
                // approximation after last iteration.

                // In infinite precision, with 0 <= eps1, eps2 <= U = 2^-HW:
                // corr_UQ1_hw = 2 - (1/b_hw + e_n) * b_hw + 2*eps1
                //             = 1 - e_n * b_hw + 2*eps1
                // x_UQ0_hw = (1/b_hw + e_n) * (1 - e_n*b_hw + 2*eps1) - eps2
                //          = 1/b_hw - e_n + 2*eps1/b_hw + e_n - e_n^2*b_hw + 2*e_n*eps1 - eps2
                //          = 1/b_hw + 2*eps1/b_hw - e_n^2*b_hw + 2*e_n*eps1 - eps2
                // e_{n+1} = -e_n^2*b_hw + 2*eps1/b_hw + 2*e_n*eps1 - eps2
                //         = 2*e_n*eps1 - (e_n^2*b_hw + eps2) + 2*eps1/b_hw
                //                        \------ >0 -------/   \-- >0 ---/
                // abs(e_{n+1}) <= 2*abs(e_n)*U + max(2*e_n^2 + U, 2 * U)
            }
            // For initial half-width iterations, U = 2^-HW
            // Let  abs(e_n)     <= u_n * U,
            // then abs(e_{n+1}) <= 2 * u_n * U^2 + max(2 * u_n^2 * U^2 + U, 2 * U)
            // u_{n+1} <= 2 * u_n * U + max(2 * u_n^2 * U + 1, 2)

            // Account for possible overflow (see above). For an overflow to occur for the
            // first time, for "ideal" corr_UQ1_hw (that is, without intermediate
            // truncation), the result of x_UQ0_hw * corr_UQ1_hw should be either maximum
            // value representable in UQ0.HW or less by 1. This means that 1/b_hw have to
            // be not below that value (see g(x) above), so it is safe to decrement just
            // once after the final iteration. On the other hand, an effective value of
            // divisor changes after this point (from b_hw to b), so adjust here.
            x_uq0_hw.wrapping_sub(1_u16)
        };

        // Error estimations for full-precision iterations are calculated just
        // as above, but with U := 2^-W and taking extra decrementing into account.
        // We need at least one such iteration.

        // Simulating operations on a twice_rep_t to perform a single final full-width
        // iteration. Using ad-hoc multiplication implementations to take advantage
        // of particular structure of operands.

        let blo: u32 = (CastInto::<u32>::cast(b_uq1)) & lo_mask;
        // x_UQ0 = x_UQ0_hw * 2^HW - 1
        // x_UQ0 * b_UQ1 = (x_UQ0_hw * 2^HW) * (b_UQ1_hw * 2^HW + blo) - b_UQ1
        //
        //   <--- higher half ---><--- lower half --->
        //   [x_UQ0_hw * b_UQ1_hw]
        // +            [  x_UQ0_hw *  blo  ]
        // -                      [      b_UQ1       ]
        // = [      result       ][.... discarded ...]
        let corr_uq1 = negate_u32(
            (x_uq0_hw as u32) * (b_uq1_hw as u32) + (((x_uq0_hw as u32) * (blo)) >> hw) - 1,
        ); // account for *possible* carry
        let lo_corr = corr_uq1 & lo_mask;
        let hi_corr = corr_uq1 >> hw;
        // x_UQ0 * corr_UQ1 = (x_UQ0_hw * 2^HW) * (hi_corr * 2^HW + lo_corr) - corr_UQ1
        let mut x_uq0: <F as Float>::Int = ((((x_uq0_hw as u32) * hi_corr) << 1)
            .wrapping_add(((x_uq0_hw as u32) * lo_corr) >> (hw - 1))
            .wrapping_sub(2))
        .cast(); // 1 to account for the highest bit of corr_UQ1 can be 1
                 // 1 to account for possible carry
                 // Just like the case of half-width iterations but with possibility
                 // of overflowing by one extra Ulp of x_UQ0.
        x_uq0 -= one;
        // ... and then traditional fixup by 2 should work

        // On error estimation:
        // abs(E_{N-1}) <=   (u_{N-1} + 2 /* due to conversion e_n -> E_n */) * 2^-HW
        //                 + (2^-HW + 2^-W))
        // abs(E_{N-1}) <= (u_{N-1} + 3.01) * 2^-HW

        // Then like for the half-width iterations:
        // With 0 <= eps1, eps2 < 2^-W
        // E_N  = 4 * E_{N-1} * eps1 - (E_{N-1}^2 * b + 4 * eps2) + 4 * eps1 / b
        // abs(E_N) <= 2^-W * [ 4 * abs(E_{N-1}) + max(2 * abs(E_{N-1})^2 * 2^W + 4, 8)) ]
        // abs(E_N) <= 2^-W * [ 4 * (u_{N-1} + 3.01) * 2^-HW + max(4 + 2 * (u_{N-1} + 3.01)^2, 8) ]
        x_uq0
    } else {
        // C is (3/4 + 1/sqrt(2)) - 1 truncated to 32 fractional bits as UQ0.n
        let c: <F as Float>::Int = (0x7504F333 << (F::BITS - 32)).cast();
        let x_uq0: <F as Float>::Int = c.wrapping_sub(b_uq1);
        // E_0 <= 3/4 - 1/sqrt(2) + 2 * 2^-32
        x_uq0
    };

    let mut x_uq0 = if USE_NATIVE_FULL_ITERATIONS {
        for _ in 0..NUMBER_OF_FULL_ITERATIONS {
            let corr_uq1: u32 = 0.wrapping_sub(
                ((CastInto::<u32>::cast(x_uq0) as u64) * (CastInto::<u32>::cast(b_uq1) as u64))
                    >> F::BITS,
            ) as u32;
            x_uq0 = ((((CastInto::<u32>::cast(x_uq0) as u64) * (corr_uq1 as u64)) >> (F::BITS - 1))
                as u32)
                .cast();
        }
        x_uq0
    } else {
        // not using native full iterations
        x_uq0
    };

    // Finally, account for possible overflow, as explained above.
    x_uq0 = x_uq0.wrapping_sub(2.cast());

    // u_n for different precisions (with N-1 half-width iterations):
    // W0 is the precision of C
    //   u_0 = (3/4 - 1/sqrt(2) + 2^-W0) * 2^HW

    // Estimated with bc:
    //   define half1(un) { return 2.0 * (un + un^2) / 2.0^hw + 1.0; }
    //   define half2(un) { return 2.0 * un / 2.0^hw + 2.0; }
    //   define full1(un) { return 4.0 * (un + 3.01) / 2.0^hw + 2.0 * (un + 3.01)^2 + 4.0; }
    //   define full2(un) { return 4.0 * (un + 3.01) / 2.0^hw + 8.0; }

    //             | f32 (0 + 3) | f32 (2 + 1)  | f64 (3 + 1)  | f128 (4 + 1)
    // u_0         | < 184224974 | < 2812.1     | < 184224974  | < 791240234244348797
    // u_1         | < 15804007  | < 242.7      | < 15804007   | < 67877681371350440
    // u_2         | < 116308    | < 2.81       | < 116308     | < 499533100252317
    // u_3         | < 7.31      |              | < 7.31       | < 27054456580
    // u_4         |             |              |              | < 80.4
    // Final (U_N) | same as u_3 | < 72         | < 218        | < 13920

    // Add 2 to U_N due to final decrement.

    let reciprocal_precision: <F as Float>::Int = 10.cast();

    // Suppose 1/b - P * 2^-W < x < 1/b + P * 2^-W
    let x_uq0 = x_uq0 - reciprocal_precision;
    // Now 1/b - (2*P) * 2^-W < x < 1/b
    // FIXME Is x_UQ0 still >= 0.5?

    let mut quotient: <F as Float>::Int = x_uq0.widen_mul(a_significand << 1).hi();
    // Now, a/b - 4*P * 2^-W < q < a/b for q=<quotient_UQ1:dummy> in UQ1.(SB+1+W).

    // quotient_UQ1 is in [0.5, 2.0) as UQ1.(SB+1),
    // adjust it to be in [1.0, 2.0) as UQ1.SB.
    let (mut residual, written_exponent) = if quotient < (implicit_bit << 1) {
        // Highest bit is 0, so just reinterpret quotient_UQ1 as UQ1.SB,
        // effectively doubling its value as well as its error estimation.
        let residual_lo = (a_significand << (significand_bits + 1)).wrapping_sub(
            (CastInto::<u32>::cast(quotient).wrapping_mul(CastInto::<u32>::cast(b_significand)))
                .cast(),
        );
        a_significand <<= 1;
        (residual_lo, written_exponent.wrapping_sub(1))
    } else {
        // Highest bit is 1 (the UQ1.(SB+1) value is in [1, 2)), convert it
        // to UQ1.SB by right shifting by 1. Least significant bit is omitted.
        quotient >>= 1;
        let residual_lo = (a_significand << significand_bits).wrapping_sub(
            (CastInto::<u32>::cast(quotient).wrapping_mul(CastInto::<u32>::cast(b_significand)))
                .cast(),
        );
        (residual_lo, written_exponent)
    };

    //drop mutability
    let quotient = quotient;

    // NB: residualLo is calculated above for the normal result case.
    //     It is re-computed on denormal path that is expected to be not so
    //     performance-sensitive.

    // Now, q cannot be greater than a/b and can differ by at most 8*P * 2^-W + 2^-SB
    // Each NextAfter() increments the floating point value by at least 2^-SB
    // (more, if exponent was incremented).
    // Different cases (<---> is of 2^-SB length, * = a/b that is shown as a midpoint):
    //   q
    //   |   | * |   |   |       |       |
    //       <--->      2^t
    //   |   |   |   |   |   *   |       |
    //               q
    // To require at most one NextAfter(), an error should be less than 1.5 * 2^-SB.
    //   (8*P) * 2^-W + 2^-SB < 1.5 * 2^-SB
    //   (8*P) * 2^-W         < 0.5 * 2^-SB
    //   P < 2^(W-4-SB)
    // Generally, for at most R NextAfter() to be enough,
    //   P < (2*R - 1) * 2^(W-4-SB)
    // For f32 (0+3): 10 < 32 (OK)
    // For f32 (2+1): 32 < 74 < 32 * 3, so two NextAfter() are required
    // For f64: 220 < 256 (OK)
    // For f128: 4096 * 3 < 13922 < 4096 * 5 (three NextAfter() are required)

    // If we have overflowed the exponent, return infinity
    if written_exponent >= max_exponent as i32 {
        return F::from_repr(inf_rep | quotient_sign);
    }

    // Now, quotient <= the correctly-rounded result
    // and may need taking NextAfter() up to 3 times (see error estimates above)
    // r = a - b * q
    let abs_result = if written_exponent > 0 {
        let mut ret = quotient & significand_mask;
        ret |= ((written_exponent as u32) << significand_bits).cast();
        residual <<= 1;
        ret
    } else {
        if (significand_bits as i32 + written_exponent) < 0 {
            return F::from_repr(quotient_sign);
        }
        let ret = quotient.wrapping_shr(negate_u32(CastInto::<u32>::cast(written_exponent)) + 1);
        residual = (CastInto::<u32>::cast(
            a_significand.wrapping_shl(
                significand_bits.wrapping_add(CastInto::<u32>::cast(written_exponent)),
            ),
        )
        .wrapping_sub(
            (CastInto::<u32>::cast(ret).wrapping_mul(CastInto::<u32>::cast(b_significand))) << 1,
        ))
        .cast();
        ret
    };
    // Round
    let abs_result = {
        residual += abs_result & one; // tie to even
                                      // The above line conditionally turns the below LT comparison into LTE

        if residual > b_significand {
            abs_result + one
        } else {
            abs_result
        }
    };
    F::from_repr(abs_result | quotient_sign)
}

pub(crate) fn div64<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
    u64: CastInto<F::Int>,
    F::Int: CastInto<u64>,
    i64: CastInto<F::Int>,
    F::Int: CastInto<i64>,
    F::Int: HInt,
{
    const NUMBER_OF_HALF_ITERATIONS: usize = 3;
    const NUMBER_OF_FULL_ITERATIONS: usize = 1;
    const USE_NATIVE_FULL_ITERATIONS: bool = false;

    let one = F::Int::ONE;
    let zero = F::Int::ZERO;
    let hw = F::BITS / 2;
    let lo_mask = u64::MAX >> hw;

    let significand_bits = F::SIGNIFICAND_BITS;
    let max_exponent = F::EXPONENT_MAX;

    let exponent_bias = F::EXPONENT_BIAS;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIGNIFICAND_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;

    #[inline(always)]
    fn negate_u64(a: u64) -> u64 {
        (<i64>::wrapping_neg(a as i64)) as u64
    }

    let a_rep = a.repr();
    let b_rep = b.repr();

    let a_exponent = (a_rep >> significand_bits) & max_exponent.cast();
    let b_exponent = (b_rep >> significand_bits) & max_exponent.cast();
    let quotient_sign = (a_rep ^ b_rep) & sign_bit;

    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;
    let mut scale = 0;

    // Detect if a or b is zero, denormal, infinity, or NaN.
    if a_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
        || b_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
    {
        let a_abs = a_rep & abs_mask;
        let b_abs = b_rep & abs_mask;

        // NaN / anything = qNaN
        if a_abs > inf_rep {
            return F::from_repr(a_rep | quiet_bit);
        }
        // anything / NaN = qNaN
        if b_abs > inf_rep {
            return F::from_repr(b_rep | quiet_bit);
        }

        if a_abs == inf_rep {
            if b_abs == inf_rep {
                // infinity / infinity = NaN
                return F::from_repr(qnan_rep);
            } else {
                // infinity / anything else = +/- infinity
                return F::from_repr(a_abs | quotient_sign);
            }
        }

        // anything else / infinity = +/- 0
        if b_abs == inf_rep {
            return F::from_repr(quotient_sign);
        }

        if a_abs == zero {
            if b_abs == zero {
                // zero / zero = NaN
                return F::from_repr(qnan_rep);
            } else {
                // zero / anything else = +/- zero
                return F::from_repr(quotient_sign);
            }
        }

        // anything else / zero = +/- infinity
        if b_abs == zero {
            return F::from_repr(inf_rep | quotient_sign);
        }

        // one or both of a or b is denormal, the other (if applicable) is a
        // normal number.  Renormalize one or both of a and b, and set scale to
        // include the necessary exponent adjustment.
        if a_abs < implicit_bit {
            let (exponent, significand) = F::normalize(a_significand);
            scale += exponent;
            a_significand = significand;
        }

        if b_abs < implicit_bit {
            let (exponent, significand) = F::normalize(b_significand);
            scale -= exponent;
            b_significand = significand;
        }
    }

    // Set the implicit significand bit.  If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;

    let written_exponent: i64 = CastInto::<u64>::cast(
        a_exponent
            .wrapping_sub(b_exponent)
            .wrapping_add(scale.cast()),
    )
    .wrapping_add(exponent_bias as u64) as i64;
    let b_uq1 = b_significand << (F::BITS - significand_bits - 1);

    // Align the significand of b as a UQ1.(n-1) fixed-point number in the range
    // [1.0, 2.0) and get a UQ0.n approximate reciprocal using a small minimax
    // polynomial approximation: x0 = 3/4 + 1/sqrt(2) - b/2.
    // The max error for this approximation is achieved at endpoints, so
    //   abs(x0(b) - 1/b) <= abs(x0(1) - 1/1) = 3/4 - 1/sqrt(2) = 0.04289...,
    // which is about 4.5 bits.
    // The initial approximation is between x0(1.0) = 0.9571... and x0(2.0) = 0.4571...

    // Then, refine the reciprocal estimate using a quadratically converging
    // Newton-Raphson iteration:
    //     x_{n+1} = x_n * (2 - x_n * b)
    //
    // Let b be the original divisor considered "in infinite precision" and
    // obtained from IEEE754 representation of function argument (with the
    // implicit bit set). Corresponds to rep_t-sized b_UQ1 represented in
    // UQ1.(W-1).
    //
    // Let b_hw be an infinitely precise number obtained from the highest (HW-1)
    // bits of divisor significand (with the implicit bit set). Corresponds to
    // half_rep_t-sized b_UQ1_hw represented in UQ1.(HW-1) that is a **truncated**
    // version of b_UQ1.
    //
    // Let e_n := x_n - 1/b_hw
    //     E_n := x_n - 1/b
    // abs(E_n) <= abs(e_n) + (1/b_hw - 1/b)
    //           = abs(e_n) + (b - b_hw) / (b*b_hw)
    //          <= abs(e_n) + 2 * 2^-HW

    // rep_t-sized iterations may be slower than the corresponding half-width
    // variant depending on the handware and whether single/double/quad precision
    // is selected.
    // NB: Using half-width iterations increases computation errors due to
    // rounding, so error estimations have to be computed taking the selected
    // mode into account!

    let mut x_uq0 = if NUMBER_OF_HALF_ITERATIONS > 0 {
        // Starting with (n-1) half-width iterations
        let b_uq1_hw: u32 =
            (CastInto::<u64>::cast(b_significand) >> (significand_bits + 1 - hw)) as u32;

        // C is (3/4 + 1/sqrt(2)) - 1 truncated to W0 fractional bits as UQ0.HW
        // with W0 being either 16 or 32 and W0 <= HW.
        // That is, C is the aforementioned 3/4 + 1/sqrt(2) constant (from which
        // b/2 is subtracted to obtain x0) wrapped to [0, 1) range.

        // HW is at least 32. Shifting into the highest bits if needed.
        let c_hw = (0x7504F333_u64 as u32).wrapping_shl(hw.wrapping_sub(32));

        // b >= 1, thus an upper bound for 3/4 + 1/sqrt(2) - b/2 is about 0.9572,
        // so x0 fits to UQ0.HW without wrapping.
        let x_uq0_hw: u32 = {
            let mut x_uq0_hw: u32 = c_hw.wrapping_sub(b_uq1_hw /* exact b_hw/2 as UQ0.HW */);
            // dbg!(x_uq0_hw);
            // An e_0 error is comprised of errors due to
            // * x0 being an inherently imprecise first approximation of 1/b_hw
            // * C_hw being some (irrational) number **truncated** to W0 bits
            // Please note that e_0 is calculated against the infinitely precise
            // reciprocal of b_hw (that is, **truncated** version of b).
            //
            // e_0 <= 3/4 - 1/sqrt(2) + 2^-W0

            // By construction, 1 <= b < 2
            // f(x)  = x * (2 - b*x) = 2*x - b*x^2
            // f'(x) = 2 * (1 - b*x)
            //
            // On the [0, 1] interval, f(0)   = 0,
            // then it increses until  f(1/b) = 1 / b, maximum on (0, 1),
            // then it decreses to     f(1)   = 2 - b
            //
            // Let g(x) = x - f(x) = b*x^2 - x.
            // On (0, 1/b), g(x) < 0 <=> f(x) > x
            // On (1/b, 1], g(x) > 0 <=> f(x) < x
            //
            // For half-width iterations, b_hw is used instead of b.
            for _ in 0..NUMBER_OF_HALF_ITERATIONS {
                // corr_UQ1_hw can be **larger** than 2 - b_hw*x by at most 1*Ulp
                // of corr_UQ1_hw.
                // "0.0 - (...)" is equivalent to "2.0 - (...)" in UQ1.(HW-1).
                // On the other hand, corr_UQ1_hw should not overflow from 2.0 to 0.0 provided
                // no overflow occurred earlier: ((rep_t)x_UQ0_hw * b_UQ1_hw >> HW) is
                // expected to be strictly positive because b_UQ1_hw has its highest bit set
                // and x_UQ0_hw should be rather large (it converges to 1/2 < 1/b_hw <= 1).
                let corr_uq1_hw: u32 =
                    0.wrapping_sub(((x_uq0_hw as u64).wrapping_mul(b_uq1_hw as u64)) >> hw) as u32;
                // dbg!(corr_uq1_hw);

                // Now, we should multiply UQ0.HW and UQ1.(HW-1) numbers, naturally
                // obtaining an UQ1.(HW-1) number and proving its highest bit could be
                // considered to be 0 to be able to represent it in UQ0.HW.
                // From the above analysis of f(x), if corr_UQ1_hw would be represented
                // without any intermediate loss of precision (that is, in twice_rep_t)
                // x_UQ0_hw could be at most [1.]000... if b_hw is exactly 1.0 and strictly
                // less otherwise. On the other hand, to obtain [1.]000..., one have to pass
                // 1/b_hw == 1.0 to f(x), so this cannot occur at all without overflow (due
                // to 1.0 being not representable as UQ0.HW).
                // The fact corr_UQ1_hw was virtually round up (due to result of
                // multiplication being **first** truncated, then negated - to improve
                // error estimations) can increase x_UQ0_hw by up to 2*Ulp of x_UQ0_hw.
                x_uq0_hw = ((x_uq0_hw as u64).wrapping_mul(corr_uq1_hw as u64) >> (hw - 1)) as u32;
                // dbg!(x_uq0_hw);
                // Now, either no overflow occurred or x_UQ0_hw is 0 or 1 in its half_rep_t
                // representation. In the latter case, x_UQ0_hw will be either 0 or 1 after
                // any number of iterations, so just subtract 2 from the reciprocal
                // approximation after last iteration.

                // In infinite precision, with 0 <= eps1, eps2 <= U = 2^-HW:
                // corr_UQ1_hw = 2 - (1/b_hw + e_n) * b_hw + 2*eps1
                //             = 1 - e_n * b_hw + 2*eps1
                // x_UQ0_hw = (1/b_hw + e_n) * (1 - e_n*b_hw + 2*eps1) - eps2
                //          = 1/b_hw - e_n + 2*eps1/b_hw + e_n - e_n^2*b_hw + 2*e_n*eps1 - eps2
                //          = 1/b_hw + 2*eps1/b_hw - e_n^2*b_hw + 2*e_n*eps1 - eps2
                // e_{n+1} = -e_n^2*b_hw + 2*eps1/b_hw + 2*e_n*eps1 - eps2
                //         = 2*e_n*eps1 - (e_n^2*b_hw + eps2) + 2*eps1/b_hw
                //                        \------ >0 -------/   \-- >0 ---/
                // abs(e_{n+1}) <= 2*abs(e_n)*U + max(2*e_n^2 + U, 2 * U)
            }
            // For initial half-width iterations, U = 2^-HW
            // Let  abs(e_n)     <= u_n * U,
            // then abs(e_{n+1}) <= 2 * u_n * U^2 + max(2 * u_n^2 * U^2 + U, 2 * U)
            // u_{n+1} <= 2 * u_n * U + max(2 * u_n^2 * U + 1, 2)

            // Account for possible overflow (see above). For an overflow to occur for the
            // first time, for "ideal" corr_UQ1_hw (that is, without intermediate
            // truncation), the result of x_UQ0_hw * corr_UQ1_hw should be either maximum
            // value representable in UQ0.HW or less by 1. This means that 1/b_hw have to
            // be not below that value (see g(x) above), so it is safe to decrement just
            // once after the final iteration. On the other hand, an effective value of
            // divisor changes after this point (from b_hw to b), so adjust here.
            x_uq0_hw.wrapping_sub(1_u32)
        };

        // Error estimations for full-precision iterations are calculated just
        // as above, but with U := 2^-W and taking extra decrementing into account.
        // We need at least one such iteration.

        // Simulating operations on a twice_rep_t to perform a single final full-width
        // iteration. Using ad-hoc multiplication implementations to take advantage
        // of particular structure of operands.
        let blo: u64 = (CastInto::<u64>::cast(b_uq1)) & lo_mask;
        // x_UQ0 = x_UQ0_hw * 2^HW - 1
        // x_UQ0 * b_UQ1 = (x_UQ0_hw * 2^HW) * (b_UQ1_hw * 2^HW + blo) - b_UQ1
        //
        //   <--- higher half ---><--- lower half --->
        //   [x_UQ0_hw * b_UQ1_hw]
        // +            [  x_UQ0_hw *  blo  ]
        // -                      [      b_UQ1       ]
        // = [      result       ][.... discarded ...]
        let corr_uq1 = negate_u64(
            (x_uq0_hw as u64) * (b_uq1_hw as u64) + (((x_uq0_hw as u64) * (blo)) >> hw) - 1,
        ); // account for *possible* carry
        let lo_corr = corr_uq1 & lo_mask;
        let hi_corr = corr_uq1 >> hw;
        // x_UQ0 * corr_UQ1 = (x_UQ0_hw * 2^HW) * (hi_corr * 2^HW + lo_corr) - corr_UQ1
        let mut x_uq0: <F as Float>::Int = ((((x_uq0_hw as u64) * hi_corr) << 1)
            .wrapping_add(((x_uq0_hw as u64) * lo_corr) >> (hw - 1))
            .wrapping_sub(2))
        .cast(); // 1 to account for the highest bit of corr_UQ1 can be 1
                 // 1 to account for possible carry
                 // Just like the case of half-width iterations but with possibility
                 // of overflowing by one extra Ulp of x_UQ0.
        x_uq0 -= one;
        // ... and then traditional fixup by 2 should work

        // On error estimation:
        // abs(E_{N-1}) <=   (u_{N-1} + 2 /* due to conversion e_n -> E_n */) * 2^-HW
        //                 + (2^-HW + 2^-W))
        // abs(E_{N-1}) <= (u_{N-1} + 3.01) * 2^-HW

        // Then like for the half-width iterations:
        // With 0 <= eps1, eps2 < 2^-W
        // E_N  = 4 * E_{N-1} * eps1 - (E_{N-1}^2 * b + 4 * eps2) + 4 * eps1 / b
        // abs(E_N) <= 2^-W * [ 4 * abs(E_{N-1}) + max(2 * abs(E_{N-1})^2 * 2^W + 4, 8)) ]
        // abs(E_N) <= 2^-W * [ 4 * (u_{N-1} + 3.01) * 2^-HW + max(4 + 2 * (u_{N-1} + 3.01)^2, 8) ]
        x_uq0
    } else {
        // C is (3/4 + 1/sqrt(2)) - 1 truncated to 64 fractional bits as UQ0.n
        let c: <F as Float>::Int = (0x7504F333 << (F::BITS - 32)).cast();
        let x_uq0: <F as Float>::Int = c.wrapping_sub(b_uq1);
        // E_0 <= 3/4 - 1/sqrt(2) + 2 * 2^-64
        x_uq0
    };

    let mut x_uq0 = if USE_NATIVE_FULL_ITERATIONS {
        for _ in 0..NUMBER_OF_FULL_ITERATIONS {
            let corr_uq1: u64 = 0.wrapping_sub(
                (CastInto::<u64>::cast(x_uq0) * (CastInto::<u64>::cast(b_uq1))) >> F::BITS,
            );
            x_uq0 = ((((CastInto::<u64>::cast(x_uq0) as u128) * (corr_uq1 as u128))
                >> (F::BITS - 1)) as u64)
                .cast();
        }
        x_uq0
    } else {
        // not using native full iterations
        x_uq0
    };

    // Finally, account for possible overflow, as explained above.
    x_uq0 = x_uq0.wrapping_sub(2.cast());

    // u_n for different precisions (with N-1 half-width iterations):
    // W0 is the precision of C
    //   u_0 = (3/4 - 1/sqrt(2) + 2^-W0) * 2^HW

    // Estimated with bc:
    //   define half1(un) { return 2.0 * (un + un^2) / 2.0^hw + 1.0; }
    //   define half2(un) { return 2.0 * un / 2.0^hw + 2.0; }
    //   define full1(un) { return 4.0 * (un + 3.01) / 2.0^hw + 2.0 * (un + 3.01)^2 + 4.0; }
    //   define full2(un) { return 4.0 * (un + 3.01) / 2.0^hw + 8.0; }

    //             | f32 (0 + 3) | f32 (2 + 1)  | f64 (3 + 1)  | f128 (4 + 1)
    // u_0         | < 184224974 | < 2812.1     | < 184224974  | < 791240234244348797
    // u_1         | < 15804007  | < 242.7      | < 15804007   | < 67877681371350440
    // u_2         | < 116308    | < 2.81       | < 116308     | < 499533100252317
    // u_3         | < 7.31      |              | < 7.31       | < 27054456580
    // u_4         |             |              |              | < 80.4
    // Final (U_N) | same as u_3 | < 72         | < 218        | < 13920

    // Add 2 to U_N due to final decrement.

    let reciprocal_precision: <F as Float>::Int = 220.cast();

    // Suppose 1/b - P * 2^-W < x < 1/b + P * 2^-W
    let x_uq0 = x_uq0 - reciprocal_precision;
    // Now 1/b - (2*P) * 2^-W < x < 1/b
    // FIXME Is x_UQ0 still >= 0.5?

    let mut quotient: <F as Float>::Int = x_uq0.widen_mul(a_significand << 1).hi();
    // Now, a/b - 4*P * 2^-W < q < a/b for q=<quotient_UQ1:dummy> in UQ1.(SB+1+W).

    // quotient_UQ1 is in [0.5, 2.0) as UQ1.(SB+1),
    // adjust it to be in [1.0, 2.0) as UQ1.SB.
    let (mut residual, written_exponent) = if quotient < (implicit_bit << 1) {
        // Highest bit is 0, so just reinterpret quotient_UQ1 as UQ1.SB,
        // effectively doubling its value as well as its error estimation.
        let residual_lo = (a_significand << (significand_bits + 1)).wrapping_sub(
            (CastInto::<u64>::cast(quotient).wrapping_mul(CastInto::<u64>::cast(b_significand)))
                .cast(),
        );
        a_significand <<= 1;
        (residual_lo, written_exponent.wrapping_sub(1))
    } else {
        // Highest bit is 1 (the UQ1.(SB+1) value is in [1, 2)), convert it
        // to UQ1.SB by right shifting by 1. Least significant bit is omitted.
        quotient >>= 1;
        let residual_lo = (a_significand << significand_bits).wrapping_sub(
            (CastInto::<u64>::cast(quotient).wrapping_mul(CastInto::<u64>::cast(b_significand)))
                .cast(),
        );
        (residual_lo, written_exponent)
    };

    //drop mutability
    let quotient = quotient;

    // NB: residualLo is calculated above for the normal result case.
    //     It is re-computed on denormal path that is expected to be not so
    //     performance-sensitive.

    // Now, q cannot be greater than a/b and can differ by at most 8*P * 2^-W + 2^-SB
    // Each NextAfter() increments the floating point value by at least 2^-SB
    // (more, if exponent was incremented).
    // Different cases (<---> is of 2^-SB length, * = a/b that is shown as a midpoint):
    //   q
    //   |   | * |   |   |       |       |
    //       <--->      2^t
    //   |   |   |   |   |   *   |       |
    //               q
    // To require at most one NextAfter(), an error should be less than 1.5 * 2^-SB.
    //   (8*P) * 2^-W + 2^-SB < 1.5 * 2^-SB
    //   (8*P) * 2^-W         < 0.5 * 2^-SB
    //   P < 2^(W-4-SB)
    // Generally, for at most R NextAfter() to be enough,
    //   P < (2*R - 1) * 2^(W-4-SB)
    // For f32 (0+3): 10 < 32 (OK)
    // For f32 (2+1): 32 < 74 < 32 * 3, so two NextAfter() are required
    // For f64: 220 < 256 (OK)
    // For f128: 4096 * 3 < 13922 < 4096 * 5 (three NextAfter() are required)

    // If we have overflowed the exponent, return infinity
    if written_exponent >= max_exponent as i64 {
        return F::from_repr(inf_rep | quotient_sign);
    }

    // Now, quotient <= the correctly-rounded result
    // and may need taking NextAfter() up to 3 times (see error estimates above)
    // r = a - b * q
    let abs_result = if written_exponent > 0 {
        let mut ret = quotient & significand_mask;
        ret |= ((written_exponent as u64) << significand_bits).cast();
        residual <<= 1;
        ret
    } else {
        if (significand_bits as i64 + written_exponent) < 0 {
            return F::from_repr(quotient_sign);
        }
        let ret =
            quotient.wrapping_shr((negate_u64(CastInto::<u64>::cast(written_exponent)) + 1) as u32);
        residual = (CastInto::<u64>::cast(
            a_significand.wrapping_shl(
                significand_bits.wrapping_add(CastInto::<u32>::cast(written_exponent)),
            ),
        )
        .wrapping_sub(
            (CastInto::<u64>::cast(ret).wrapping_mul(CastInto::<u64>::cast(b_significand))) << 1,
        ))
        .cast();
        ret
    };
    // Round
    let abs_result = {
        residual += abs_result & one; // tie to even
                                      // conditionally turns the below LT comparison into LTE
        if residual > b_significand {
            abs_result + one
        } else {
            abs_result
        }
    };
    F::from_repr(abs_result | quotient_sign)
}

pub(crate) fn mul<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
    F::Int: HInt,
{
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;

    let bits = F::BITS;
    let significand_bits = F::SIGNIFICAND_BITS;
    let max_exponent = F::EXPONENT_MAX;

    let exponent_bias = F::EXPONENT_BIAS;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIGNIFICAND_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;
    let exponent_bits = F::EXPONENT_BITS;

    let a_rep = a.repr();
    let b_rep = b.repr();

    let a_exponent = (a_rep >> significand_bits) & max_exponent.cast();
    let b_exponent = (b_rep >> significand_bits) & max_exponent.cast();
    let product_sign = (a_rep ^ b_rep) & sign_bit;

    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;
    let mut scale = 0;

    // Detect if a or b is zero, denormal, infinity, or NaN.
    if a_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
        || b_exponent.wrapping_sub(one) >= (max_exponent - 1).cast()
    {
        let a_abs = a_rep & abs_mask;
        let b_abs = b_rep & abs_mask;

        // NaN + anything = qNaN
        if a_abs > inf_rep {
            return F::from_repr(a_rep | quiet_bit);
        }
        // anything + NaN = qNaN
        if b_abs > inf_rep {
            return F::from_repr(b_rep | quiet_bit);
        }

        if a_abs == inf_rep {
            if b_abs != zero {
                // infinity * non-zero = +/- infinity
                return F::from_repr(a_abs | product_sign);
            } else {
                // infinity * zero = NaN
                return F::from_repr(qnan_rep);
            }
        }

        if b_abs == inf_rep {
            if a_abs != zero {
                // infinity * non-zero = +/- infinity
                return F::from_repr(b_abs | product_sign);
            } else {
                // infinity * zero = NaN
                return F::from_repr(qnan_rep);
            }
        }

        // zero * anything = +/- zero
        if a_abs == zero {
            return F::from_repr(product_sign);
        }

        // anything * zero = +/- zero
        if b_abs == zero {
            return F::from_repr(product_sign);
        }

        // one or both of a or b is denormal, the other (if applicable) is a
        // normal number.  Renormalize one or both of a and b, and set scale to
        // include the necessary exponent adjustment.
        if a_abs < implicit_bit {
            let (exponent, significand) = F::normalize(a_significand);
            scale += exponent;
            a_significand = significand;
        }

        if b_abs < implicit_bit {
            let (exponent, significand) = F::normalize(b_significand);
            scale += exponent;
            b_significand = significand;
        }
    }

    // Or in the implicit significand bit.  (If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.)
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;

    // Get the significand of a*b.  Before multiplying the significands, shift
    // one of them left to left-align it in the field.  Thus, the product will
    // have (exponentBits + 2) integral digits, all but two of which must be
    // zero.  Normalizing this result is just a conditional left-shift by one
    // and bumping the exponent accordingly.
    let (mut product_low, mut product_high) = a_significand
        .widen_mul(b_significand << exponent_bits)
        .lo_hi();

    let a_exponent_i32: i32 = a_exponent.cast();
    let b_exponent_i32: i32 = b_exponent.cast();
    let mut product_exponent: i32 = a_exponent_i32
        .wrapping_add(b_exponent_i32)
        .wrapping_add(scale)
        .wrapping_sub(exponent_bias as i32);

    // Normalize the significand, adjust exponent if needed.
    if (product_high & implicit_bit) != zero {
        product_exponent = product_exponent.wrapping_add(1);
    } else {
        product_high = (product_high << 1) | (product_low >> (bits - 1));
        product_low <<= 1;
    }

    // If we have overflowed the type, return +/- infinity.
    if product_exponent >= max_exponent as i32 {
        return F::from_repr(inf_rep | product_sign);
    }

    if product_exponent <= 0 {
        // Result is denormal before rounding
        //
        // If the result is so small that it just underflows to zero, return
        // a zero of the appropriate sign.  Mathematically there is no need to
        // handle this case separately, but we make it a special case to
        // simplify the shift logic.
        let shift = one.wrapping_sub(product_exponent.cast()).cast();
        if shift >= bits {
            return F::from_repr(product_sign);
        }

        // Otherwise, shift the significand of the result so that the round
        // bit is the high bit of productLo.
        if shift < bits {
            let sticky = product_low << (bits - shift);
            product_low = product_high << (bits - shift) | product_low >> shift | sticky;
            product_high >>= shift;
        } else if shift < (2 * bits) {
            let sticky = product_high << (2 * bits - shift) | product_low;
            product_low = product_high >> (shift - bits) | sticky;
            product_high = zero;
        } else {
            product_high = zero;
        }
    } else {
        // Result is normal before rounding; insert the exponent.
        product_high &= significand_mask;
        product_high |= product_exponent.cast() << significand_bits;
    }

    // Insert the sign of the result:
    product_high |= product_sign;

    // Final rounding.  The final result may overflow to infinity, or underflow
    // to zero, but those are the correct results in those cases.  We use the
    // default IEEE-754 round-to-nearest, ties-to-even rounding mode.
    if product_low > sign_bit {
        product_high += one;
    }

    if product_low == sign_bit {
        product_high += product_high & one;
    }

    F::from_repr(product_high)
}

/// Returns `a` raised to the power `b`
pub(crate) fn powif(a: f32, b: i32) -> f32 {
    let mut a = a;
    let recip = b < 0;
    let mut pow = Int::abs_diff(b, 0);
    let mut mul = 1.0_f32;
    loop {
        if (pow & 1) != 0 {
            mul = crate::compiler_builtins::mul(mul, a);
        }
        pow >>= 1;
        if pow == 0 {
            break;
        }
        a = crate::compiler_builtins::mul(a, a);
    }
    if recip {
        crate::compiler_builtins::div32(1.0_f32, mul)
    } else {
        mul
    }
}

/// Returns `a` raised to the power `b`
pub(crate) fn powi(a: f64, b: i32) -> f64 {
    let mut a = a;
    let recip = b < 0;
    let mut pow = Int::abs_diff(b, 0);
    let mut mul = 1.0_f64;
    loop {
        if (pow & 1) != 0 {
            mul = crate::compiler_builtins::mul(mul, a);
        }
        pow >>= 1;
        if pow == 0 {
            break;
        }
        a = crate::compiler_builtins::mul(a, a);
    }
    if recip {
        crate::compiler_builtins::div64(1.0_f64, mul)
    } else {
        mul
    }
}

pub(crate) fn sub<F: Float>(a: F, b: F) -> F {
    a - b
}
