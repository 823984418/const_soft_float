# Rust float-point in constant context

Float-point code is from `compiler_builtins = "0.1.87"` and the code has been rewrite so that make it work in a constant context.

If there is any inconsistent behavior, please open an issues.

features:
* `no_std`
* `const_trait_impl`
* `const_mut_refs`

work in `stable`:
```
const fn const_f32_add(a: f32, b: f32) -> f32 {
    SoftF32(a).add(SoftF32(b)).to_f32()
}
```


with `const_trait_impl` usage:
```
const fn const_f32_add(a: f32, b: f32) -> f32 {
    (SoftF32(a) + SoftF32(b)).to_f32()
}
```

with `const_mut_refs` usage:
```
const fn const_f32_add(a: f32, b: f32) -> f32 {
    let mut x = SoftF32(a);
    x += SoftF32(b);
    x.to_f32()
}
```
