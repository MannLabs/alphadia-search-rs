//! Sorting the values of a feature column: float order keys and an LSD radix sort.

/// An integer key of `v` in `f32::total_cmp` order: the bits of a positive float are in order,
/// those of a negative one in reverse order.
pub(super) fn order_key(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}

pub(super) fn from_order_key(key: u32) -> f32 {
    f32::from_bits(if key >> 31 == 1 {
        key & !(1 << 31)
    } else {
        !key
    })
}

/// LSD radix sort, several times faster than a comparison sort on the millions of keys of a
/// feature.
pub(super) fn radix_sort(mut keys: Vec<u32>) -> Vec<u32> {
    const BITS: usize = 11;
    const BUCKETS: usize = 1 << BITS;
    const PASSES: usize = 32usize.div_ceil(BITS);
    let digit = |key: u32, pass: usize| (key >> (pass * BITS)) as usize & (BUCKETS - 1);
    let mut counts = vec![[0usize; BUCKETS]; PASSES];
    for &key in &keys {
        for (pass, counts) in counts.iter_mut().enumerate() {
            counts[digit(key, pass)] += 1;
        }
    }
    let mut sorted = vec![0u32; keys.len()];
    for (pass, counts) in counts.iter_mut().enumerate() {
        if counts.contains(&keys.len()) {
            continue; // all keys share this digit
        }
        let mut offset = 0;
        for count in counts.iter_mut() {
            (*count, offset) = (offset, offset + *count);
        }
        for &key in &keys {
            let slot = &mut counts[digit(key, pass)];
            sorted[*slot] = key;
            *slot += 1;
        }
        std::mem::swap(&mut keys, &mut sorted);
    }
    keys
}
