use super::*;
use approx::assert_relative_eq;
use numpy::ndarray::{arr1, arr2};

#[test]
fn test_axis_dot_product_basic_case() {
    let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let weights = vec![0.5, 1.5];
    let result = axis_dot_product(&array, &weights);

    // Correct expected values: 0.5*1.0 + 1.5*4.0 = 0.5 + 6.0 = 6.5
    //                          0.5*2.0 + 1.5*5.0 = 1.0 + 7.5 = 8.5
    //                          0.5*3.0 + 1.5*6.0 = 1.5 + 9.0 = 10.5
    let expected = arr1(&[6.5, 8.5, 10.5]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn test_axis_dot_product_single_row() {
    let array = arr2(&[[1.0, 2.0, 3.0]]);
    let weights = vec![2.0];
    let result = axis_dot_product(&array, &weights);

    let expected = arr1(&[2.0, 4.0, 6.0]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
fn test_axis_dot_product_all_zeros() {
    let array = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
    let weights = vec![1.0, 1.0];
    let result = axis_dot_product(&array, &weights);

    let expected = arr1(&[0.0, 0.0]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

#[test]
#[should_panic]
fn test_axis_dot_product_dimension_mismatch() {
    // Should panic because weights.len() != array.dim().0
    let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let weights = vec![0.5];
    let _ = axis_dot_product(&array, &weights);
}

#[test]
fn test_axis_log_sum_basic() {
    let array = arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let result = axis_log_sum(&array);

    // Expected: ln(0.0+1.0) + ln(3.0+1.0) = 0.0 + ln(4.0) = 1.386...
    //           ln(1.0+1.0) + ln(4.0+1.0) = ln(2.0) + ln(5.0) = 0.693... + 1.609... = 2.302...
    //           ln(2.0+1.0) + ln(5.0+1.0) = ln(3.0) + ln(6.0) = 1.098... + 1.791... = 2.889...
    let expected = arr1(&[1.3862944, 2.3025851, 2.8903718]);
    for (a, b) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-5);
    }
}

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






