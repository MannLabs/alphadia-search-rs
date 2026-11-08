//! `f32` buffers aligned to a cache line, so the 64-byte vector loads and stores of the kernels
//! never straddle two lines (`Vec<f32>` is only 16-byte aligned, numpy arrays likewise).

use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([f32; 16]);

/// A fixed-length, zero-initialised `f32` buffer starting on a 64-byte boundary.
#[derive(Clone)]
pub struct AlignedBuf {
    lines: Vec<Line>,
    len: usize,
}

impl AlignedBuf {
    pub fn zeros(len: usize) -> Self {
        AlignedBuf {
            lines: vec![Line([0.0; 16]); len.div_ceil(16)],
            len,
        }
    }

    pub fn from_slice(values: &[f32]) -> Self {
        let mut buf = Self::zeros(values.len());
        buf.copy_from_slice(values);
        buf
    }
}

impl Deref for AlignedBuf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: `lines` holds at least `len` contiguous, initialised f32 values
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast(), self.len) }
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as in deref, and `self` is borrowed mutably
        unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast(), self.len) }
    }
}
