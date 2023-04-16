#![cfg(test)]

use crate::sf32::{fuzz_test_op, fuzz_test_op_2, fuzz_test_op_2_other};
use const_soft_float::soft_f32::SoftF32;

#[test]
fn fuzz_add() {
    fuzz_test_op_2(
        SoftF32::add,
        crate::compiler_builtins::add,
        crate::Argument::First,
        Some("add"),
    );
    fuzz_test_op_2(
        SoftF32::add,
        crate::compiler_builtins::add,
        crate::Argument::Second,
        Some("add"),
    );
}

#[test]
fn fuzz_sub() {
    fuzz_test_op_2(
        SoftF32::sub,
        crate::compiler_builtins::sub,
        crate::Argument::First,
        Some("sub"),
    );
    fuzz_test_op_2(
        SoftF32::sub,
        crate::compiler_builtins::sub,
        crate::Argument::Second,
        Some("sub"),
    );
}

#[test]
fn fuzz_div() {
    fuzz_test_op_2(
        SoftF32::div,
        crate::compiler_builtins::div32,
        crate::Argument::First,
        Some("div"),
    );
    fuzz_test_op_2(
        SoftF32::div,
        crate::compiler_builtins::div32,
        crate::Argument::Second,
        Some("div"),
    );
}

#[test]
fn fuzz_mul() {
    fuzz_test_op_2(
        SoftF32::mul,
        crate::compiler_builtins::mul,
        crate::Argument::First,
        Some("mul"),
    );
    fuzz_test_op_2(
        SoftF32::mul,
        crate::compiler_builtins::mul,
        crate::Argument::Second,
        Some("mul"),
    );
}

#[test]
fn fuzz_pow() {
    fuzz_test_op_2_other(SoftF32::powi, crate::compiler_builtins::pow, Some("pow"))
}

#[test]
fn fuzz_sqrt() {
    fuzz_test_op(SoftF32::sqrt, libm::sqrtf, Some("sqrt"))
}

#[test]
fn fuzz_round() {
    fuzz_test_op(SoftF32::round, f32::round, Some("round"))
}

#[test]
fn fuzz_trunc() {
    fuzz_test_op(SoftF32::trunc, f32::trunc, Some("trunc"));
}

#[test]
fn fuzz_copysign() {
    fuzz_test_op_2(
        SoftF32::copysign,
        libm::copysignf,
        crate::Argument::First,
        Some("copysign"),
    );
    fuzz_test_op_2(
        SoftF32::copysign,
        libm::copysignf,
        crate::Argument::Second,
        Some("copysign"),
    );
}
