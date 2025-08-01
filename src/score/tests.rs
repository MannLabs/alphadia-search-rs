use super::*;
use approx::assert_relative_eq;
use numpy::ndarray::{arr1, arr2};

#[test]
fn test_axis_log_dot_product_basic() {
    let array = arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let weights = vec![0.5, 1.5];
    let result = axis_log_dot_product(&array, &weights);

    // Expected: 0.5*ln(0.0+1.0) + 1.5*ln(3.0+1.0) = 0.5*0.0 + 1.5*ln(4.0) = 0.0 + 1.5*1.386... = 2.079...
    //           0.5*ln(1.0+1.0) + 1.5*ln(4.0+1.0) = 0.5*ln(2.0) + 1.5*ln(5.0) = 0.5*0.693... + 1.5*1.609... = 0.346... + 2.413... = 2.759...
    //           0.5*ln(2.0+1.0) + 1.5*ln(5.0+1.0) = 0.5*ln(3.0) + 1.5*ln(6.0) = 0.5*1.098... + 1.5*1.791... = 0.549... + 2.686... = 3.235...
    let expected = arr1(&[2.0794415, 2.7607305, 3.2369454]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn test_axis_sqrt_dot_product_basic() {
    let array = arr2(&[[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
    let weights = vec![0.5, 1.5];
    let result = axis_sqrt_dot_product(&array, &weights);

    // Expected: 0.5*sqrt(0.0) + 1.5*sqrt(9.0) = 0.0 + 1.5*3.0 = 4.5
    //           0.5*sqrt(1.0) + 1.5*sqrt(16.0) = 0.5*1.0 + 1.5*4.0 = 0.5 + 6.0 = 6.5
    //           0.5*sqrt(4.0) + 1.5*sqrt(25.0) = 0.5*2.0 + 1.5*5.0 = 1.0 + 7.5 = 8.5
    let expected = arr1(&[4.5, 6.5, 8.5]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn test_axis_sqrt_dot_product_negative_values() {
    let array = arr2(&[[-1.0, 0.0, 4.0], [9.0, -4.0, 25.0]]);
    let weights = vec![0.5, 1.5];
    let result = axis_sqrt_dot_product(&array, &weights);

    // Expected: 0.5*sqrt(0.0) + 1.5*sqrt(9.0) = 0.0 + 1.5*3.0 = 4.5
    //           0.5*sqrt(0.0) + 1.5*sqrt(0.0) = 0.0 + 0.0 = 0.0
    //           0.5*sqrt(4.0) + 1.5*sqrt(25.0) = 0.5*2.0 + 1.5*5.0 = 1.0 + 7.5 = 8.5
    let expected = arr1(&[4.5, 0.0, 8.5]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}
