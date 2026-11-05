use super::*;
use pyo3::prelude::*;

#[test]
fn test_candidate_collection_to_arrays_dtypes_and_order() {
    pyo3::Python::initialize();
    let candidate = Candidate {
        precursor_idx: 42,
        rank: 7,
        score: 1.5,
        scan_center: 101,
        scan_start: 100,
        scan_stop: 102,
        cycle_center: 201,
        cycle_start: 200,
        cycle_stop: 202,
    };
    let collection = CandidateCollection::from_vec(vec![candidate]);

    Python::attach(|py| {
        let (
            precursor_idxs,
            ranks,
            scores,
            scan_center,
            scan_start,
            scan_stop,
            cycle_center,
            cycle_start,
            cycle_stop,
        ) = collection.to_arrays(py).expect("to_arrays should succeed");

        let precursor_idxs: Vec<u64> = precursor_idxs.extract(py).unwrap();
        let ranks: Vec<u64> = ranks.extract(py).unwrap();
        let scores: Vec<f32> = scores.extract(py).unwrap();
        let scan_center: Vec<u64> = scan_center.extract(py).unwrap();
        let scan_start: Vec<u64> = scan_start.extract(py).unwrap();
        let scan_stop: Vec<u64> = scan_stop.extract(py).unwrap();
        let cycle_center: Vec<u64> = cycle_center.extract(py).unwrap();
        let cycle_start: Vec<u64> = cycle_start.extract(py).unwrap();
        let cycle_stop: Vec<u64> = cycle_stop.extract(py).unwrap();

        assert_eq!(precursor_idxs.len(), 1);
        assert_eq!(ranks.len(), 1);
        assert_eq!(scores.len(), 1);
        assert_eq!(scan_center.len(), 1);
        assert_eq!(scan_start.len(), 1);
        assert_eq!(scan_stop.len(), 1);
        assert_eq!(cycle_center.len(), 1);
        assert_eq!(cycle_start.len(), 1);
        assert_eq!(cycle_stop.len(), 1);

        assert_eq!(precursor_idxs[0], 42u64);
        assert_eq!(ranks[0], 7u64);
        assert!((scores[0] - 1.5).abs() < 1e-6);
        assert_eq!(scan_center[0], 101u64);
        assert_eq!(scan_start[0], 100u64);
        assert_eq!(scan_stop[0], 102u64);
        assert_eq!(cycle_center[0], 201u64);
        assert_eq!(cycle_start[0], 200u64);
        assert_eq!(cycle_stop[0], 202u64);
    });
}

#[test]
fn test_candidate_collection_from_arrays_roundtrip() {
    pyo3::Python::initialize();
    let precursor_idxs = vec![1u64, 2u64];
    let ranks = vec![10u64, 20u64];
    let scores = vec![0.1f32, 0.2f32];
    let scan_center = vec![11u64, 21u64];
    let scan_start = vec![12u64, 22u64];
    let scan_stop = vec![13u64, 23u64];
    let cycle_center = vec![14u64, 24u64];
    let cycle_start = vec![15u64, 25u64];
    let cycle_stop = vec![16u64, 26u64];

    let collection = CandidateCollection::from_arrays(
        precursor_idxs.clone(),
        ranks.clone(),
        scores.clone(),
        scan_center.clone(),
        scan_start.clone(),
        scan_stop.clone(),
        cycle_center.clone(),
        cycle_start.clone(),
        cycle_stop.clone(),
    )
    .expect("from_arrays should succeed");

    assert_eq!(collection.len(), 2);
    assert!(!collection.is_empty());

    Python::attach(|py| {
        let (p, r, s, sc, ss, so, cc, cs, co) = collection.to_arrays(py).unwrap();

        let p: Vec<u64> = p.extract(py).unwrap();
        let r: Vec<u64> = r.extract(py).unwrap();
        let s: Vec<f32> = s.extract(py).unwrap();
        let sc: Vec<u64> = sc.extract(py).unwrap();
        let ss: Vec<u64> = ss.extract(py).unwrap();
        let so: Vec<u64> = so.extract(py).unwrap();
        let cc: Vec<u64> = cc.extract(py).unwrap();
        let cs: Vec<u64> = cs.extract(py).unwrap();
        let co: Vec<u64> = co.extract(py).unwrap();

        assert_eq!(p, precursor_idxs);
        assert_eq!(r, ranks);
        assert_eq!(s, scores);
        assert_eq!(sc, scan_center);
        assert_eq!(ss, scan_start);
        assert_eq!(so, scan_stop);
        assert_eq!(cc, cycle_center);
        assert_eq!(cs, cycle_start);
        assert_eq!(co, cycle_stop);
    });
}
