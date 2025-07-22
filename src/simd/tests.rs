use super::*;
use serial_test::serial;

// comment
#[test]
#[serial]
fn test_natural_backend_selection() {
    // Test natural backend selection without overrides
    clear_backend();

    let selected_backend = test_backend();

    #[cfg(target_arch = "aarch64")]
    {
        if NEON.is_available() {
            assert_eq!(
                selected_backend, "neon",
                "Should select NEON on aarch64 with NEON support"
            );
            println!("✓ NEON backend correctly selected on aarch64");
        } else {
            assert_eq!(
                selected_backend, "scalar",
                "Should fall back to scalar when NEON unavailable"
            );
            println!("⚠️  NEON not available, fell back to scalar backend");
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        assert_eq!(
            selected_backend, "scalar",
            "Should select scalar backend on non-aarch64"
        );
        println!("⚠️  Non-aarch64 architecture - SIMD testing limited to scalar backend");
    }
}

#[test]
#[serial]
fn test_scalar_backend_override() {
    // Test forcing scalar backend
    clear_backend();

    set_backend("scalar").expect("Should be able to set scalar backend");
    assert_eq!(get_current_backend(), "scalar");
    assert_eq!(test_backend(), "scalar");

    println!("✓ Successfully forced scalar backend");

    clear_backend();
}

#[test]
#[serial]
fn test_neon_backend_override() {
    // Test forcing NEON backend (aarch64 only)
    clear_backend();

    #[cfg(target_arch = "aarch64")]
    {
        if NEON.is_available() {
            set_backend("neon").expect("Should be able to set NEON backend when available");
            assert_eq!(get_current_backend(), "neon");
            assert_eq!(test_backend(), "neon");
            println!("✓ Successfully forced NEON backend");
        } else {
            let result = set_backend("neon");
            assert!(
                result.is_err(),
                "Should fail to set unavailable NEON backend"
            );
            println!("⚠️  NEON backend correctly rejected when not available");
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let result = set_backend("neon");
        assert!(
            result.is_err(),
            "Should fail to set NEON backend on non-aarch64"
        );
        println!("⚠️  NEON backend correctly rejected on non-aarch64");
    }

    clear_backend();
}

#[test]
#[serial]
fn test_optimal_vs_current_backend() {
    // Test that get_optimal_simd_backend() ignores overrides while get_current_backend() respects them
    clear_backend();

    #[cfg(target_arch = "aarch64")]
    {
        if NEON.is_available() {
            // Force scalar when NEON is optimal
            set_backend("scalar").expect("Should be able to set scalar backend");

            assert_eq!(
                get_current_backend(),
                "scalar",
                "Current should respect override"
            );
            assert_eq!(
                get_optimal_simd_backend(),
                "neon",
                "Optimal should ignore override"
            );

            println!("✓ get_optimal_simd_backend() correctly ignores user override");
            println!("✓ get_current_backend() correctly respects user override");
        } else {
            println!("⚠️  Cannot test override differences - NEON not available");
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Both should return scalar on non-aarch64
        set_backend("scalar").expect("Should be able to set scalar backend");

        assert_eq!(get_current_backend(), "scalar");
        assert_eq!(get_optimal_simd_backend(), "scalar");

        println!("⚠️  Non-aarch64: both functions return scalar (expected)");
    }

    clear_backend();
}

#[test]
#[serial]
fn test_invalid_backend_error_handling() {
    // Test error handling for invalid backend names
    clear_backend();

    let result = set_backend("invalid_backend");
    assert!(result.is_err(), "Should fail to set invalid backend");

    // Natural selection should still work after failed set
    let backend = get_current_backend();

    #[cfg(target_arch = "aarch64")]
    {
        if NEON.is_available() {
            assert_eq!(
                backend, "neon",
                "Should maintain natural selection after invalid backend"
            );
        } else {
            assert_eq!(
                backend, "scalar",
                "Should maintain natural selection after invalid backend"
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        assert_eq!(
            backend, "scalar",
            "Should maintain natural selection after invalid backend"
        );
    }

    println!("✓ Invalid backend name correctly rejected, natural selection maintained");

    clear_backend();
}

#[test]
#[serial]
fn test_architecture_info() {
    // Emit useful information about the testing environment
    #[cfg(target_arch = "aarch64")]
    {
        if NEON.is_available() {
            println!("✓ Running on aarch64 with NEON support - full SIMD testing available");
        } else {
            println!("⚠️  Running on aarch64 but NEON is not available (unusual)");
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("⚠️  Running on non-aarch64 architecture");
        println!("⚠️  SIMD testing is limited - only scalar backend available");
        println!("⚠️  For complete SIMD testing, run on aarch64 (Apple Silicon/ARM64)");
    }

    // This test always passes - it's just for information
    assert!(true);
}
