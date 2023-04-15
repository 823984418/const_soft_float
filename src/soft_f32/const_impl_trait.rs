use crate::soft_f32::SoftF32;

type F = SoftF32;

impl const From<f32> for F {
    fn from(value: f32) -> Self {
        F::from_f32(value)
    }
}

impl const PartialEq<Self> for F {
    fn eq(&self, other: &Self) -> bool {
        match self.cmp(*other) {
            Some(core::cmp::Ordering::Equal) => true,
            _ => false,
        }
    }
}

impl const PartialOrd for F {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.cmp(*other)
    }
}

impl const core::ops::Add for F {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        F::add(self, rhs)
    }
}

impl const core::ops::Sub for F {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        F::sub(self, rhs)
    }
}

impl const core::ops::Mul for F {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        F::mul(self, rhs)
    }
}

impl const core::ops::Div for F {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        F::div(self, rhs)
    }
}

#[cfg(feature = "const_mut_refs")]
impl const core::ops::AddAssign for F {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[cfg(not(feature = "const_mut_refs"))]
impl core::ops::AddAssign for F {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[cfg(feature = "const_mut_refs")]
impl const core::ops::SubAssign for F {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(not(feature = "const_mut_refs"))]
impl core::ops::SubAssign for F {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
