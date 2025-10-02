use super::*;

#[test]
fn test_idf_creation_and_calculation() {
    let library_mz = vec![100.0, 100.0];
    let idf = InverseDocumentFrequency::new(&library_mz);

    let idf_values = idf.get_idf(&[100.0, 200.0]);

    // 100.0 appears twice (df=2): IDF = ln(2/2) = 0
    assert!((idf_values[0] - 0.0).abs() < 1e-6);

    // 200.0 not in library (df=1): IDF = ln(2/1) = ln(2)
    assert!((idf_values[1] - 2.0_f32.ln()).abs() < 1e-6);
}

#[test]
fn test_empty_library() {
    let library_mz: Vec<f32> = vec![];
    let idf = InverseDocumentFrequency::new(&library_mz);

    assert_eq!(idf.total_fragments, 0.0);

    let query_mz = vec![100.0, 200.0, 300.0];
    let idf_values = idf.get_idf(&query_mz);

    // Should return all ones when library is empty
    assert_eq!(idf_values.len(), 3);
    assert_eq!(idf_values, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_single_fragment() {
    let library_mz = vec![150.0];
    let idf = InverseDocumentFrequency::new(&library_mz);

    let idf_values = idf.get_idf(&[150.0]);

    // Single fragment (df=1): IDF = ln(1/1) = 0
    assert!((idf_values[0] - 0.0).abs() < 1e-6);
}
