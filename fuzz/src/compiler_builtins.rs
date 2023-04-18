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
{
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;

    // let bits = F::BITS;
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

    // Or in the implicit significand bit.  (If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.)
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;
    let mut quotient_exponent: i32 = CastInto::<i32>::cast(a_exponent)
        .wrapping_sub(CastInto::<i32>::cast(b_exponent))
        .wrapping_add(scale);

    // Align the significand of b as a Q31 fixed-point number in the range
    // [1, 2.0) and get a Q32 approximate reciprocal using a small minimax
    // polynomial approximation: reciprocal = 3/4 + 1/sqrt(2) - b/2.  This
    // is accurate to about 3.5 binary digits.
    let q31b = CastInto::<u32>::cast(b_significand << 8.cast());
    let mut reciprocal = (0x7504f333u32).wrapping_sub(q31b);

    // Now refine the reciprocal estimate using a Newton-Raphson iteration:
    //
    //     x1 = x0 * (2 - x0 * b)
    //
    // This doubles the number of correct binary digits in the approximation
    // with each iteration, so after three iterations, we have about 28 binary
    // digits of accuracy.

    let mut correction: u32 =
        negate_u32(((reciprocal as u64).wrapping_mul(q31b as u64) >> 32) as u32);
    reciprocal = ((reciprocal as u64).wrapping_mul(correction as u64) >> 31) as u32;
    correction = negate_u32(((reciprocal as u64).wrapping_mul(q31b as u64) >> 32) as u32);
    reciprocal = ((reciprocal as u64).wrapping_mul(correction as u64) >> 31) as u32;
    correction = negate_u32(((reciprocal as u64).wrapping_mul(q31b as u64) >> 32) as u32);
    reciprocal = ((reciprocal as u64).wrapping_mul(correction as u64) >> 31) as u32;

    // Exhaustive testing shows that the error in reciprocal after three steps
    // is in the interval [-0x1.f58108p-31, 0x1.d0e48cp-29], in line with our
    // expectations.  We bump the reciprocal by a tiny value to force the error
    // to be strictly positive (in the range [0x1.4fdfp-37,0x1.287246p-29], to
    // be specific).  This also causes 1/1 to give a sensible approximation
    // instead of zero (due to overflow).
    reciprocal = reciprocal.wrapping_sub(2);

    // The numerical reciprocal is accurate to within 2^-28, lies in the
    // interval [0x1.000000eep-1, 0x1.fffffffcp-1], and is strictly smaller
    // than the true reciprocal of b.  Multiplying a by this reciprocal thus
    // gives a numerical q = a/b in Q24 with the following properties:
    //
    //    1. q < a/b
    //    2. q is in the interval [0x1.000000eep-1, 0x1.fffffffcp0)
    //    3. the error in q is at most 2^-24 + 2^-27 -- the 2^24 term comes
    //       from the fact that we truncate the product, and the 2^27 term
    //       is the error in the reciprocal of b scaled by the maximum
    //       possible value of a.  As a consequence of this error bound,
    //       either q or nextafter(q) is the correctly rounded
    let mut quotient = (a_significand << 1).widen_mul(reciprocal.cast()).hi();

    // Two cases: quotient is in [0.5, 1.0) or quotient is in [1.0, 2.0).
    // In either case, we are going to compute a residual of the form
    //
    //     r = a - q*b
    //
    // We know from the construction of q that r satisfies:
    //
    //     0 <= r < ulp(q)*b
    //
    // if r is greater than 1/2 ulp(q)*b, then q rounds up.  Otherwise, we
    // already have the correct result.  The exact halfway case cannot occur.
    // We also take this time to right shift quotient if it falls in the [1,2)
    // range and adjust the exponent accordingly.
    let residual = if quotient < (implicit_bit << 1) {
        quotient_exponent = quotient_exponent.wrapping_sub(1);
        (a_significand << (significand_bits + 1)).wrapping_sub(quotient.wrapping_mul(b_significand))
    } else {
        quotient >>= 1;
        (a_significand << significand_bits).wrapping_sub(quotient.wrapping_mul(b_significand))
    };

    let written_exponent = quotient_exponent.wrapping_add(exponent_bias as i32);

    if written_exponent >= max_exponent as i32 {
        // If we have overflowed the exponent, return infinity.
        return F::from_repr(inf_rep | quotient_sign);
    } else if written_exponent < 1 {
        // Flush denormals to zero.  In the future, it would be nice to add
        // code to round them correctly.
        return F::from_repr(quotient_sign);
    } else {
        let round = ((residual << 1) > b_significand) as u32;
        // Clear the implicit bits
        let mut abs_result = quotient & significand_mask;
        // Insert the exponent
        abs_result |= written_exponent.cast() << significand_bits;
        // Round
        abs_result = abs_result.wrapping_add(round.cast());
        // Insert the sign and return
        return F::from_repr(abs_result | quotient_sign);
    }
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
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;

    // let bits = F::BITS;
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
    // let exponent_bits = F::EXPONENT_BITS;

    #[inline(always)]
    fn negate_u32(a: u32) -> u32 {
        (<i32>::wrapping_neg(a as i32)) as u32
    }

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

    // Or in the implicit significand bit.  (If we fell through from the
    // denormal path it was already set by normalize( ), but setting it twice
    // won't hurt anything.)
    a_significand |= implicit_bit;
    b_significand |= implicit_bit;
    let mut quotient_exponent: i32 = CastInto::<i32>::cast(a_exponent)
        .wrapping_sub(CastInto::<i32>::cast(b_exponent))
        .wrapping_add(scale);

    // Align the significand of b as a Q31 fixed-point number in the range
    // [1, 2.0) and get a Q32 approximate reciprocal using a small minimax
    // polynomial approximation: reciprocal = 3/4 + 1/sqrt(2) - b/2.  This
    // is accurate to about 3.5 binary digits.
    let q31b = CastInto::<u32>::cast(b_significand >> 21.cast());
    let mut recip32 = (0x7504f333u32).wrapping_sub(q31b);

    // Now refine the reciprocal estimate using a Newton-Raphson iteration:
    //
    //     x1 = x0 * (2 - x0 * b)
    //
    // This doubles the number of correct binary digits in the approximation
    // with each iteration, so after three iterations, we have about 28 binary
    // digits of accuracy.

    let mut correction32: u32 =
        negate_u32(((recip32 as u64).wrapping_mul(q31b as u64) >> 32) as u32);
    recip32 = ((recip32 as u64).wrapping_mul(correction32 as u64) >> 31) as u32;
    correction32 = negate_u32(((recip32 as u64).wrapping_mul(q31b as u64) >> 32) as u32);
    recip32 = ((recip32 as u64).wrapping_mul(correction32 as u64) >> 31) as u32;
    correction32 = negate_u32(((recip32 as u64).wrapping_mul(q31b as u64) >> 32) as u32);
    recip32 = ((recip32 as u64).wrapping_mul(correction32 as u64) >> 31) as u32;

    // recip32 might have overflowed to exactly zero in the preceeding
    // computation if the high word of b is exactly 1.0.  This would sabotage
    // the full-width final stage of the computation that follows, so we adjust
    // recip32 downward by one bit.
    recip32 = recip32.wrapping_sub(1);

    // We need to perform one more iteration to get us to 56 binary digits;
    // The last iteration needs to happen with extra precision.
    let q63blo = CastInto::<u32>::cast(b_significand << 11.cast());

    let correction: u64 = negate_u64(
        (recip32 as u64)
            .wrapping_mul(q31b as u64)
            .wrapping_add((recip32 as u64).wrapping_mul(q63blo as u64) >> 32),
    );
    let c_hi = (correction >> 32) as u32;
    let c_lo = correction as u32;
    let mut reciprocal: u64 = (recip32 as u64)
        .wrapping_mul(c_hi as u64)
        .wrapping_add((recip32 as u64).wrapping_mul(c_lo as u64) >> 32);

    // We already adjusted the 32-bit estimate, now we need to adjust the final
    // 64-bit reciprocal estimate downward to ensure that it is strictly smaller
    // than the infinitely precise exact reciprocal.  Because the computation
    // of the Newton-Raphson step is truncating at every step, this adjustment
    // is small; most of the work is already done.
    reciprocal = reciprocal.wrapping_sub(2);

    // The numerical reciprocal is accurate to within 2^-56, lies in the
    // interval [0.5, 1.0), and is strictly smaller than the true reciprocal
    // of b.  Multiplying a by this reciprocal thus gives a numerical q = a/b
    // in Q53 with the following properties:
    //
    //    1. q < a/b
    //    2. q is in the interval [0.5, 2.0)
    //    3. the error in q is bounded away from 2^-53 (actually, we have a
    //       couple of bits to spare, but this is all we need).

    // We need a 64 x 64 multiply high to compute q, which isn't a basic
    // operation in C, so we need to be a little bit fussy.
    // let mut quotient: F::Int = ((((reciprocal as u64)
    //     .wrapping_mul(CastInto::<u32>::cast(a_significand << 1) as u64))
    //     >> 32) as u32)
    //     .cast();

    // We need a 64 x 64 multiply high to compute q, which isn't a basic
    // operation in C, so we need to be a little bit fussy.
    let mut quotient = (a_significand << 2).widen_mul(reciprocal.cast()).hi();

    // Two cases: quotient is in [0.5, 1.0) or quotient is in [1.0, 2.0).
    // In either case, we are going to compute a residual of the form
    //
    //     r = a - q*b
    //
    // We know from the construction of q that r satisfies:
    //
    //     0 <= r < ulp(q)*b
    //
    // if r is greater than 1/2 ulp(q)*b, then q rounds up.  Otherwise, we
    // already have the correct result.  The exact halfway case cannot occur.
    // We also take this time to right shift quotient if it falls in the [1,2)
    // range and adjust the exponent accordingly.
    let residual = if quotient < (implicit_bit << 1) {
        quotient_exponent = quotient_exponent.wrapping_sub(1);
        (a_significand << (significand_bits + 1)).wrapping_sub(quotient.wrapping_mul(b_significand))
    } else {
        quotient >>= 1;
        (a_significand << significand_bits).wrapping_sub(quotient.wrapping_mul(b_significand))
    };

    let written_exponent = quotient_exponent.wrapping_add(exponent_bias as i32);

    if written_exponent >= max_exponent as i32 {
        // If we have overflowed the exponent, return infinity.
        return F::from_repr(inf_rep | quotient_sign);
    } else if written_exponent < 1 {
        // Flush denormals to zero.  In the future, it would be nice to add
        // code to round them correctly.
        return F::from_repr(quotient_sign);
    } else {
        let round = ((residual << 1) > b_significand) as u32;
        // Clear the implicit bits
        let mut abs_result = quotient & significand_mask;
        // Insert the exponent
        abs_result |= written_exponent.cast() << significand_bits;
        // Round
        abs_result = abs_result.wrapping_add(round.cast());
        // Insert the sign and return
        return F::from_repr(abs_result | quotient_sign);
    }
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
