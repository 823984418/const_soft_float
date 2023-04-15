use crate::soft_f64::SoftF64;

type F = SoftF64;

impl From<f64> for F {
    fn from(value: f64) -> Self {
        F::from_f64(value)
    }
}

impl PartialEq<Self> for F {
    fn eq(&self, other: &Self) -> bool {
        match self.cmp(*other) {
            Some(core::cmp::Ordering::Equal) => true,
            _ => false,
        }
    }
}

impl PartialOrd for F {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.cmp(*other)
    }
}

impl core::ops::Add for F {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        F::add(self, rhs)
    }
}

impl core::ops::Sub for F {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        F::sub(self, rhs)
    }
}

impl core::ops::Mul for F {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        F::mul(self, rhs)
    }
}

impl core::ops::Div for F {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        F::div(self, rhs)
    }
}

impl core::ops::AddAssign for F {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for F {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
