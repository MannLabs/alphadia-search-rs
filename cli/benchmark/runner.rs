use std::time::Instant;

pub struct BenchmarkResult {
    pub test_case: &'static str,
    pub tag: &'static str,
    pub implementation: &'static str,
    pub time_seconds: f64,
    pub speedup: f64,
    pub avg_rel_error: f32,
    pub max_rel_error: f32,
    pub status: &'static str,
}

pub trait BenchmarkCase {
    fn name(&self) -> &'static str;
    fn tag(&self) -> &'static str;
    fn run(&self, n_rounds: usize) -> Vec<BenchmarkResult>;
}

pub type ImplList<D, O> = Vec<(&'static str, fn(&D) -> O)>;

pub struct TypedBenchmarkCase<D, O> {
    pub name: &'static str,
    pub tag: &'static str,
    pub generate: Box<dyn Fn() -> D>,
    pub implementations: ImplList<D, O>,
    pub compute_error: fn(&O, &O) -> (f32, f32),
    pub max_error_tolerance: f32,
}

impl<D, O> BenchmarkCase for TypedBenchmarkCase<D, O> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn tag(&self) -> &'static str {
        self.tag
    }

    fn run(&self, n_rounds: usize) -> Vec<BenchmarkResult> {
        let datasets: Vec<D> = (0..n_rounds).map(|_| (self.generate)()).collect();

        // Warmup on first dataset
        for (_, imp_fn) in &self.implementations {
            let _ = imp_fn(&datasets[0]);
        }

        let n_impls = self.implementations.len();
        let mut times = vec![vec![0.0f64; n_rounds]; n_impls];
        let mut avg_errors = vec![vec![0.0f32; n_rounds]; n_impls];
        let mut max_errors = vec![vec![0.0f32; n_rounds]; n_impls];

        for (round, data) in datasets.iter().enumerate() {
            // Run reference (first) implementation
            let start = Instant::now();
            let ref_output = (self.implementations[0].1)(data);
            times[0][round] = start.elapsed().as_secs_f64();

            for (impl_idx, (_, imp_fn)) in self.implementations.iter().enumerate().skip(1) {
                let start = Instant::now();
                let output = imp_fn(data);
                times[impl_idx][round] = start.elapsed().as_secs_f64();

                let (avg_e, max_e) = (self.compute_error)(&ref_output, &output);
                avg_errors[impl_idx][round] = avg_e;
                max_errors[impl_idx][round] = max_e;
            }
        }

        let scalar_avg_time: f64 = times[0].iter().sum::<f64>() / n_rounds as f64;

        let mut results = Vec::with_capacity(n_impls);
        for (impl_idx, (impl_name, _)) in self.implementations.iter().enumerate() {
            let avg_time: f64 = times[impl_idx].iter().sum::<f64>() / n_rounds as f64;
            let speedup = scalar_avg_time / avg_time;

            let avg_err: f32 = if impl_idx == 0 {
                0.0
            } else {
                avg_errors[impl_idx].iter().sum::<f32>() / n_rounds as f32
            };
            let worst_max: f32 = if impl_idx == 0 {
                0.0
            } else {
                max_errors[impl_idx].iter().cloned().fold(0.0f32, f32::max)
            };

            let status = if impl_idx == 0 || worst_max < self.max_error_tolerance {
                "PASS"
            } else {
                "FAIL"
            };

            results.push(BenchmarkResult {
                test_case: self.name,
                tag: self.tag,
                implementation: impl_name,
                time_seconds: avg_time,
                speedup,
                avg_rel_error: avg_err,
                max_rel_error: worst_max,
                status,
            });
        }

        results
    }
}
