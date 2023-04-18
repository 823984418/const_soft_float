use crate::{soft_f32::SoftF32, soft_f64::SoftF64};

/// https://github.com/rust-lang/libm/blob/4c8a973741c014b11ce7f1477693a3e5d4ef9609/src/math/k_sinf.rs
pub(crate) const fn k_sinf(x: SoftF64) -> SoftF32 {
    const S1: SoftF64 = SoftF64(-0.166666666416265235595); /* -0x15555554cbac77.0p-55 */
    const S2: SoftF64 = SoftF64(0.0083333293858894631756); /*  0x111110896efbb2.0p-59 */
    const S3: SoftF64 = SoftF64(-0.000198393348360966317347); /* -0x1a00f9e2cae774.0p-65 */
    const S4: SoftF64 = SoftF64(0.0000027183114939898219064); /*  0x16cd878c3b46a7.0p-71 */
    let z = x.mul(x);
    let w = z.mul(z);
    let r = S3.add(z.mul(S4));
    let s = z.mul(x);
    SoftF32((x.add(s.mul(S1.add(z.mul(S2))))).add(s.mul(w).mul(r)).0 as f32)
}
