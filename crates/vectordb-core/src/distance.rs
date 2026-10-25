//! Distance metrics and SIMD kernels.

#![deny(unsafe_code)]

use crate::error::{Error, Result};

/// Supported distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Cosine distance = 1 - cosine_similarity. Vectors are normalized on insert.
    Cosine,
    /// Squared Euclidean (L2) distance.
    L2,
    /// Negative dot product (so smaller = more similar). Stored as -dot.
    Dot,
}

impl Metric {
    /// Parse from a string.
    pub fn parse(s: &str) -> Result<Self> {
        <Self as std::str::FromStr>::from_str(s)
    }

    /// Canonical name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::L2 => "l2",
            Metric::Dot => "dot",
        }
    }
}

impl std::str::FromStr for Metric {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Metric::Cosine),
            "l2" | "euclidean" => Ok(Metric::L2),
            "dot" | "inner_product" => Ok(Metric::Dot),
            _ => Err(Error::UnknownMetric(s.to_string())),
        }
    }
}

/// Compute L2 norm.
pub fn norm(v: &[f32]) -> f32 {
    sqrtf(dot(v, v))
}

/// Dot product (scalar-ish via simd module).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::distance::simd::dot(a, b)
}

/// Squared L2 distance.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    crate::distance::simd::l2_squared(a, b)
}

/// Cosine distance between possibly-unnormalized vectors.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - d / (na * nb)
}

/// Normalize a vector in place to unit length.
pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n == 0.0 {
        return;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Distance according to metric. For cosine, assumes vectors already normalized
/// (so distance = 1 - dot). For dot metric, returns -dot (smaller = closer).
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Error::DimMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let v = match metric {
        Metric::Cosine => 1.0 - dot(a, b),
        Metric::L2 => l2_squared(a, b),
        Metric::Dot => -dot(a, b),
    };
    Ok(v)
}

fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}

pub mod simd {
    //! SIMD kernels with runtime feature detection and scalar fallback.
    #![allow(unsafe_code)]

    /// Dot product with runtime dispatch.
    #[allow(unreachable_code)]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: feature detected at runtime.
                return unsafe { dot_avx512(a, b) };
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: feature detected at runtime.
                return unsafe { dot_avx2(a, b) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64; use it.
            // SAFETY: NEON is always available on aarch64.
            return unsafe { dot_neon(a, b) };
        }
        dot_scalar(a, b)
    }

    /// Squared L2 distance with runtime dispatch.
    #[allow(unreachable_code)]
    pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { l2_avx512(a, b) };
            }
            if is_x86_feature_detected!("avx2") {
                return unsafe { l2_avx2(a, b) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { l2_neon(a, b) };
        }
        l2_scalar(a, b)
    }

    // ---- scalar fallbacks ----
    pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut s = 0.0f32;
        for i in 0..n {
            s += a[i] * b[i];
        }
        s
    }

    pub fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut s = 0.0f32;
        for i in 0..n {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }

    // ---- x86_64 AVX2 ----
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let n = a.len().min(b.len());
        let mut i = 0;
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            // SAFETY: bounds checked by loop condition.
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        // horizontal sum
        let mut tmp = [0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s: f32 = tmp.iter().sum();
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let n = a.len().min(b.len());
        let mut i = 0;
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
            i += 8;
        }
        let mut tmp = [0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s: f32 = tmp.iter().sum();
        while i < n {
            let d = a[i] - b[i];
            s += d * d;
            i += 1;
        }
        s
    }

    // ---- x86_64 AVX-512 ----
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let n = a.len().min(b.len());
        let mut i = 0;
        let mut acc = _mm512_setzero_ps();
        while i + 16 <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
        }
        let mut s = _mm512_reduce_add_ps(acc);
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let n = a.len().min(b.len());
        let mut i = 0;
        let mut acc = _mm512_setzero_ps();
        while i + 16 <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += 16;
        }
        let mut s = _mm512_reduce_add_ps(acc);
        while i < n {
            let d = a[i] - b[i];
            s += d * d;
            i += 1;
        }
        s
    }

    // ---- aarch64 NEON ----
    #[cfg(target_arch = "aarch64")]
    unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;
        let n = a.len().min(b.len());
        let mut i = 0;
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }
        let mut s = vaddvq_f32(acc);
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn l2_neon(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;
        let n = a.len().min(b.len());
        let mut i = 0;
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let d = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, d, d);
            i += 4;
        }
        let mut s = vaddvq_f32(acc);
        while i < n {
            let d = a[i] - b[i];
            s += d * d;
            i += 1;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn smoke() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    proptest! {
        #[test]
        fn simd_matches_scalar_dot_varied_dim(n in proptest::sample::select(vec![1usize,3,7,15,17,100,129])) {
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            for i in 0..n {
                a[i] = (i as f32) * 0.1 - 5.0;
                b[i] = (i as f32) * 0.2 + 1.0;
            }
            let s = simd::dot_scalar(&a, &b);
            let v = simd::dot(&a, &b);
            prop_assert!((s - v).abs() < 1e-2 + 1e-3 * s.abs(), "dot mismatch {} vs {}", s, v);
        }
        #[test]
        fn simd_matches_scalar_l2_varied_dim(n in proptest::sample::select(vec![1usize,3,7,15,17,100,129])) {
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            for i in 0..n {
                a[i] = (i as f32) * 0.1 - 5.0;
                b[i] = (i as f32) * 0.2 + 1.0;
            }
            let s = simd::l2_scalar(&a, &b);
            let v = simd::l2_squared(&a, &b);
            prop_assert!((s - v).abs() < 1e-2 + 1e-3 * s.abs(), "l2 mismatch {} vs {}", s, v);
        }
        #[test]
        fn simd_matches_scalar_dot(a in proptest::collection::vec(-10.0f32..10.0, 0..64),
                                    b in proptest::collection::vec(-10.0f32..10.0, 0..64)) {
            // pad to same length
            let n = a.len().min(b.len());
            let a = &a[..n];
            let b = &b[..n];
            let s = simd::dot_scalar(a, b);
            let v = simd::dot(a, b);
            prop_assert!((s - v).abs() < 1e-3 + 1e-3 * s.abs());
        }
        #[test]
        fn simd_matches_scalar_l2(a in proptest::collection::vec(-10.0f32..10.0, 0..64),
                                    b in proptest::collection::vec(-10.0f32..10.0, 0..64)) {
            let n = a.len().min(b.len());
            let a = &a[..n];
            let b = &b[..n];
            let s = simd::l2_scalar(a, b);
            let v = simd::l2_squared(a, b);
            prop_assert!((s - v).abs() < 1e-3 + 1e-3 * s.abs());
        }
    }
}
