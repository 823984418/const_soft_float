use crate::soft_f64::SoftF64;
use core::cmp::Ordering;

type F = SoftF64;

type FInt = u64;
type FSignedInt = i64;

const UNORDERED: Option<Ordering> = None;
const EQUAL: Option<Ordering> = Some(Ordering::Equal);
const GREATER: Option<Ordering> = Some(Ordering::Greater);
const LESS: Option<Ordering> = Some(Ordering::Less);

pub(crate) const fn cmp(a: F, b: F) -> Option<Ordering> {
    let one: FInt = 1;
    let zero: FInt = 0;
    let szero: FSignedInt = 0;

    let sign_bit = F::SIGN_MASK as FInt;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep = exponent_mask;

    let a_rep = a.repr();
    let b_rep = b.repr();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    // If either a or b is NaN, they are unordered.
    if a_abs > inf_rep || b_abs > inf_rep {
        return UNORDERED;
    }

    // If a and b are both zeros, they are equal.
    if a_abs | b_abs == zero {
        return EQUAL;
    }

    let a_srep = a.signed_repr();
    let b_srep = b.signed_repr();

    // If at least one of a and b is positive, we get the same result comparing
    // a and b as signed integers as we would with a fp_ting-point compare.
    if a_srep & b_srep >= szero {
        if a_srep < b_srep {
            LESS
        } else if a_srep == b_srep {
            EQUAL
        } else {
            GREATER
        }
        // Otherwise, both are negative, so we need to flip the sense of the
        // comparison to get the correct result.  (This assumes a twos- or ones-
        // complement integer representation; if integers are represented in a
        // sign-magnitude representation, then this flip is incorrect).
    } else if a_srep > b_srep {
        LESS
    } else if a_srep == b_srep {
        EQUAL
    } else {
        GREATER
    }
}
