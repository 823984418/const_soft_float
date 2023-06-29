# Rust float-point in constant context

Floating-point code is from `compiler_builtins = "0.1.94"` and `libm = "0.2.6"`, and has been rewritten for use in a constant context. 

Some fuzzing of operations is performed to ensure correctness of the port, but please open an issue if there is any inconsistent behavior.

Features:
* `no_std`
* `const_trait_impl`
* `const_mut_refs`

On `stable`:
```
const fn const_f32_add(a: f32, b: f32) -> f32 {
    SoftF32(a).add(SoftF32(b)).to_f32()
}
```


On `nightly` with `const_trait_impl` usage:
```
const fn const_f32_add(a: f32, b: f32) -> f32 {
    (SoftF32(a) + SoftF32(b)).to_f32()
}
```

On `nightly` with `const_mut_refs` usage:
```
const fn const_f32_add(a: f32, b: f32) -> f32 {
    let mut x = SoftF32(a);
    x += SoftF32(b);
    x.to_f32()
}
```
