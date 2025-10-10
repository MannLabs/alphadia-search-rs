pub fn trapezoidal_integration(rt: &[f32], intensity: &[f32]) -> f32 {
    if rt.len() != intensity.len() || rt.len() < 2 {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 1..rt.len() {
        let delta_rt = rt[i] - rt[i - 1];
        let sum_intensity = intensity[i] + intensity[i - 1];
        area += delta_rt * sum_intensity / 2.0;
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trapezoidal_integration_simple() {
        let rt = vec![1.0, 2.0, 3.0, 4.0];
        let intensity = vec![10.0, 20.0, 5.0, 0.0];
        // (1*(10+20)/2) + (1*(20+5)/2) + (1*(5+0)/2) = 15 + 12.5 + 2.5 = 30.0
        assert_eq!(trapezoidal_integration(&rt, &intensity), 30.0);
    }

    #[test]
    fn test_trapezoidal_integration_two_points() {
        let rt = vec![1.0, 2.0];
        let intensity = vec![10.0, 20.0];
        // (1*(10+20)/2) = 15.0
        assert_eq!(trapezoidal_integration(&rt, &intensity), 15.0);
    }

    #[test]
    fn test_trapezoidal_integration_single_point() {
        let rt = vec![1.0];
        let intensity = vec![10.0];
        assert_eq!(trapezoidal_integration(&rt, &intensity), 0.0);
    }

    #[test]
    fn test_trapezoidal_integration_empty() {
        let rt = vec![];
        let intensity = vec![];
        assert_eq!(trapezoidal_integration(&rt, &intensity), 0.0);
    }

    #[test]
    fn test_trapezoidal_integration_uneven_spacing() {
        let rt = vec![1.0, 3.0, 4.0, 6.0];
        let intensity = vec![10.0, 20.0, 5.0, 0.0];
        // (2*(10+20)/2) + (1*(20+5)/2) + (2*(5+0)/2) = 30 + 12.5 + 5 = 47.5
        assert_eq!(trapezoidal_integration(&rt, &intensity), 47.5);
    }
}