use core::cmp::Ordering;

use crate::soft_f64::SoftF64;

pub(crate) const fn eq(l: SoftF64, r: SoftF64) -> bool {
    if let Some(ord) = l.cmp(r) {
        match ord {
            Ordering::Equal => true,
            _ => false,
        }
    } else {
        panic!("Failed to compare values");
    }
}

pub(crate) const fn gt(l: SoftF64, r: SoftF64) -> bool {
    if let Some(ord) = l.cmp(r) {
        match ord {
            Ordering::Greater => true,
            _ => false,
        }
    } else {
        panic!("Failed to compare values");
    }
}

pub(crate) const fn ge(l: SoftF64, r: SoftF64) -> bool {
    if let Some(ord) = l.cmp(r) {
        match ord {
            Ordering::Less => false,
            _ => true,
        }
    } else {
        panic!("Failed to compare values");
    }
}
