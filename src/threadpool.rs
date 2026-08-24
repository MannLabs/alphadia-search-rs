use once_cell::sync::Lazy;
use std::sync::Mutex;

static THREADPOOL_INIT: Lazy<Mutex<Option<usize>>> = Lazy::new(|| Mutex::new(None));

/// Set the number of threads for Rayon's global thread pool.
///
/// This function is idempotent: calling it multiple times with the same thread count
/// will succeed. However, attempting to change the thread count after initialization
/// will return an error.
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
/// * `Ok(())` if successful or if already initialized with the same thread count
/// * `Err(String)` if attempting to change thread count or if configuration failed
pub fn set_num_threads(num_threads: Option<usize>) -> Result<(), String> {
    let mut initialized = THREADPOOL_INIT
        .lock()
        .map_err(|e| format!("Failed to acquire lock: {e}"))?;

    if let Some(n) = num_threads {
        if n == 0 {
            return Err("Number of threads must be greater than 0".to_string());
        }
    }

    let requested_threads = num_threads.unwrap_or_else(rayon::current_num_threads);

    if let Some(current_threads) = *initialized {
        if current_threads == requested_threads {
            return Ok(());
        }
        return Err(format!(
            "Thread pool already initialized with {current_threads} threads, cannot change to {requested_threads}"
        ));
    }

    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = num_threads {
        builder = builder.num_threads(n);
    }

    builder
        .build_global()
        .map_err(|e| format!("Failed to configure thread pool: {e}"))?;

    let actual_threads = rayon::current_num_threads();
    *initialized = Some(actual_threads);
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
