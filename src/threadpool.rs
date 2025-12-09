use once_cell::sync::Lazy;
use std::sync::Mutex;

static THREADPOOL_INIT: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

/// Set the number of threads for Rayon's global thread pool.
///
/// This function must be called before any parallel operations are performed.
/// If called after parallelization has started, it will return an error.
///
/// # Arguments
///
/// * `num_threads` - The number of threads to use. If None, uses all available CPUs.
///
/// # Returns
///
/// * `Ok(())` if successful
/// * `Err(String)` if the thread pool was already initialized or configuration failed
pub fn set_num_threads(num_threads: Option<usize>) -> Result<(), String> {
    let mut initialized = THREADPOOL_INIT
        .lock()
        .map_err(|e| format!("Failed to acquire lock: {e}"))?;

    if *initialized {
        return Err(
            "Thread pool already initialized. set_num_threads() must be called immediately after import, before any other operations.".to_string()
        );
    }

    let mut builder = rayon::ThreadPoolBuilder::new();

    if let Some(n) = num_threads {
        if n == 0 {
            return Err("Number of threads must be greater than 0".to_string());
        }
        builder = builder.num_threads(n);
    }

    builder
        .build_global()
        .map_err(|e| format!("Failed to configure thread pool: {e}"))?;

    *initialized = true;
    Ok(())
}

/// Get the current number of threads in use by Rayon.
///
/// # Returns
///
/// The number of threads in the current thread pool.
pub fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_num_threads() {
        let num_threads = get_num_threads();
        assert!(num_threads > 0, "Should have at least one thread");
    }
}
