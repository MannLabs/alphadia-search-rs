use super::*;
use crate::kernel::GaussianKernel;
use numpy::ndarray::Array2;
use rand::prelude::*;

#[test]
fn test_convolution_similarity() {
    let n_points = 100;
    let n_fragments = 6;
    let num_arrays = 10;
    let kernel_width = 20;
    let tolerance = 1e-5;

    let kernel = GaussianKernel::new(2.0, 1.0, kernel_width, 1.0);

    let mut rng = rand::rng();
    let mut arrays = Vec::with_capacity(num_arrays);
    for _ in 0..num_arrays {
        let mut arr = Array2::<f32>::zeros((n_fragments, n_points));
        for i in 0..n_fragments {
            for j in 0..n_points {
                arr[[i, j]] = rng.random_range(0.0..1.0);
            }
        }
        arrays.push(arr);
    }

    let implementations: Vec<(&str, fn(&GaussianKernel, &Array2<f32>) -> Array2<f32>)> = vec![
        ("Branching", benchmark_padded_convolution_branching),
        (
            "Branching+SIMD",
            benchmark_padded_convolution_branching_simd,
        ),
        ("Symmetric+SIMD", benchmark_symmetric_kernel_simd),
    ];

    for arr in &arrays {
        let reference = benchmark_padded_convolution(&kernel, arr);

        for (name, implementation) in &implementations {
            let result = implementation(&kernel, arr);

            let mut max_diff = 0.0f32;
            let (nf, np) = reference.dim();

            for f in 0..nf {
                for p in 0..np {
                    let diff = (reference[[f, p]] - result[[f, p]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            assert!(
                max_diff <= tolerance,
                "{name} differs from Original by {max_diff}, exceeding tolerance {tolerance}"
            );
        }
    }
}
