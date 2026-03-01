use rand::prelude::*;

use alphadia_search_rs::constants::FragmentType;
#[cfg(target_arch = "aarch64")]
use alphadia_search_rs::peak_group_scoring::utils::calculate_hyperscore_neon;
use alphadia_search_rs::peak_group_scoring::utils::{
    calculate_hyperscore_branchless, calculate_hyperscore_lut, calculate_hyperscore_naive,
    calculate_hyperscore_weighted,
};

use super::runner::{BenchmarkCase, ImplList, TypedBenchmarkCase};

const TAG: &str = "hyperscore";

struct HyperscoreData {
    fragment_types: Vec<Vec<u8>>,
    fragment_intensities: Vec<Vec<f32>>,
    matched_masks: Vec<Vec<bool>>,
}

fn gen_hyperscore_data(n_fragments: usize, n_candidates: usize) -> HyperscoreData {
    let mut rng = rand::rng();
    let types_pool = [FragmentType::B, FragmentType::Y, FragmentType::A];

    let mut fragment_types = Vec::with_capacity(n_candidates);
    let mut fragment_intensities = Vec::with_capacity(n_candidates);
    let mut matched_masks = Vec::with_capacity(n_candidates);

    for _ in 0..n_candidates {
        let types: Vec<u8> = (0..n_fragments)
            .map(|_| types_pool[rng.random_range(0..3)])
            .collect();
        let intensities: Vec<f32> = (0..n_fragments)
            .map(|_| rng.random_range(0.0..1000.0))
            .collect();
        let mask: Vec<bool> = (0..n_fragments).map(|_| rng.random_bool(0.7)).collect();

        fragment_types.push(types);
        fragment_intensities.push(intensities);
        matched_masks.push(mask);
    }

    HyperscoreData {
        fragment_types,
        fragment_intensities,
        matched_masks,
    }
}

#[allow(clippy::ptr_arg)]
fn hyperscore_error(reference: &Vec<f32>, candidate: &Vec<f32>) -> (f32, f32) {
    let mut sum_rel = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..reference.len() {
        let rel = if reference[i].abs() > 1e-12 {
            (reference[i] - candidate[i]).abs() / reference[i].abs()
        } else {
            (reference[i] - candidate[i]).abs()
        };
        sum_rel += rel;
        max_rel = max_rel.max(rel);
    }
    let avg_rel = sum_rel / reference.len() as f32;
    (avg_rel, max_rel)
}

fn run_naive(data: &HyperscoreData) -> Vec<f32> {
    data.fragment_types
        .iter()
        .zip(data.fragment_intensities.iter())
        .zip(data.matched_masks.iter())
        .map(|((types, intensities), mask)| calculate_hyperscore_naive(types, intensities, mask))
        .collect()
}

fn run_stirling(data: &HyperscoreData) -> Vec<f32> {
    data.fragment_types
        .iter()
        .zip(data.fragment_intensities.iter())
        .zip(data.matched_masks.iter())
        .map(|((types, intensities), mask)| {
            calculate_hyperscore_weighted(types, intensities, mask, None)
        })
        .collect()
}

fn run_lut(data: &HyperscoreData) -> Vec<f32> {
    data.fragment_types
        .iter()
        .zip(data.fragment_intensities.iter())
        .zip(data.matched_masks.iter())
        .map(|((types, intensities), mask)| {
            calculate_hyperscore_lut(types, intensities, mask, None)
        })
        .collect()
}

fn run_branchless(data: &HyperscoreData) -> Vec<f32> {
    data.fragment_types
        .iter()
        .zip(data.fragment_intensities.iter())
        .zip(data.matched_masks.iter())
        .map(|((types, intensities), mask)| {
            calculate_hyperscore_branchless(types, intensities, mask, None)
        })
        .collect()
}

#[cfg(target_arch = "aarch64")]
fn run_neon(data: &HyperscoreData) -> Vec<f32> {
    data.fragment_types
        .iter()
        .zip(data.fragment_intensities.iter())
        .zip(data.matched_masks.iter())
        .map(|((types, intensities), mask)| {
            calculate_hyperscore_neon(types, intensities, mask, None)
        })
        .collect()
}

type HyperscoreImpls = ImplList<HyperscoreData, Vec<f32>>;

fn hyperscore_impls() -> HyperscoreImpls {
    let mut impls: HyperscoreImpls = vec![
        ("Naive", run_naive as fn(&HyperscoreData) -> Vec<f32>),
        ("Stirling", run_stirling),
        ("LUT", run_lut),
        ("Branchless", run_branchless),
    ];
    #[cfg(target_arch = "aarch64")]
    impls.push(("NEON-v2", run_neon));
    impls
}

fn make_hyperscore(
    name: &'static str,
    n_fragments: usize,
    n_candidates: usize,
) -> TypedBenchmarkCase<HyperscoreData, Vec<f32>> {
    TypedBenchmarkCase {
        name,
        tag: TAG,
        generate: Box::new(move || gen_hyperscore_data(n_fragments, n_candidates)),
        implementations: hyperscore_impls(),
        compute_error: hyperscore_error,
        max_error_tolerance: 0.05,
    }
}

pub fn test_cases() -> Vec<Box<dyn BenchmarkCase>> {
    vec![
        Box::new(make_hyperscore("hyperscore_12x1k", 12, 1_000)),
        Box::new(make_hyperscore("hyperscore_48x1k", 48, 1_000)),
        Box::new(make_hyperscore("hyperscore_12x10k", 12, 10_000)),
        Box::new(make_hyperscore("hyperscore_48x10k", 48, 10_000)),
        Box::new(make_hyperscore("hyperscore_12x100k", 12, 100_000)),
        Box::new(make_hyperscore("hyperscore_48x100k", 48, 100_000)),
    ]
}
