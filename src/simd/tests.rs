use super::*;
use serial_test::serial;

#[test]
#[serial]
fn test_backend_selection_without_override() {
    // Test natural backend selection
    clear_backend(); // Clear any previous override
    let result = test_backend();
    
    // On aarch64 with NEON support, should select neon
    // Otherwise, should fall back to scalar
    #[cfg(target_arch = "aarch64")]
    {
        // Check if NEON is actually available
        if NEON.is_available() {
            assert_eq!(result, "aarch64_neon", "Should select NEON backend on aarch64 with NEON support");
            println!("✓ NEON backend correctly selected on aarch64");
        } else {
            assert_eq!(result, "aarch64_scalar", "Should fall back to scalar when NEON unavailable");
            println!("⚠️  NEON not available, fell back to scalar backend");
        }
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert_eq!(result, "aarch64_scalar", "Should select scalar backend on non-aarch64");
        println!("⚠️  Non-aarch64 architecture detected - SIMD testing limited to scalar backend");
    }
}

#[test]
#[serial]
fn test_force_scalar_backend() {
    // Force scalar backend via set_backend API
    set_backend("scalar").expect("Should be able to set scalar backend");
    let result = test_backend();
    
    assert_eq!(result, "aarch64_scalar", "Should use scalar backend when set via API");
    println!("✓ Successfully forced scalar backend via set_backend() API");
    
    // Clean up
    clear_backend();
}

#[test]
#[serial]
fn test_force_neon_backend() {
    // Only test NEON forcing on aarch64
    #[cfg(target_arch = "aarch64")]
    {
        // Force NEON backend via set_backend API
        if NEON.is_available() {
            set_backend("neon").expect("Should be able to set NEON backend when available");
            let result = test_backend();
            assert_eq!(result, "aarch64_neon", "Should use NEON backend when set and available");
            println!("✓ Successfully forced NEON backend via set_backend() API");
        } else {
            // Test that setting unavailable NEON fails gracefully
            let result = set_backend("neon");
            assert!(result.is_err(), "Should fail to set unavailable NEON backend");
            println!("⚠️  NEON backend correctly rejected when not available");
        }
        
        // Clean up
        clear_backend();
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("⚠️  Skipping NEON force test - not on aarch64 architecture");
    }
}

#[test]
#[serial]
fn test_invalid_backend_fallback() {
    // Try to set an invalid backend
    let set_result = set_backend("invalid_backend");
    assert!(set_result.is_err(), "Should fail to set invalid backend");
    
    // Backend should fall back to natural selection
    clear_backend();
    let result = test_backend();
    
    // Should fall back to the best available backend (same as no override)
            #[cfg(target_arch = "aarch64")]
        {
            if NEON.is_available() {
                assert_eq!(result, "aarch64_neon", "Should fall back to NEON when invalid backend specified");
            } else {
                assert_eq!(result, "aarch64_scalar", "Should fall back to scalar when invalid backend specified");
            }
        }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert_eq!(result, "aarch64_scalar", "Should fall back to scalar when invalid backend specified");
    }
    
    println!("✓ Invalid backend name correctly ignored, fell back to best available");
    
    // Clean up
    clear_backend();
}

#[test]
#[serial]
fn test_optimal_simd_backend_function() {
    // Test the optimal SIMD backend function with natural backend selection
    clear_backend(); // Ensure natural selection
    
    let name = get_optimal_simd_backend();
    
    #[cfg(target_arch = "aarch64")]
    {
        if NEON.is_available() {
            assert_eq!(name, "neon");
            println!("✓ Optimal SIMD backend correctly reports NEON");
        } else {
            assert_eq!(name, "scalar");
            println!("⚠️  Optimal SIMD backend correctly reports scalar fallback");
        }
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert_eq!(name, "scalar");
        println!("⚠️  Optimal SIMD backend correctly reports scalar on non-aarch64 architecture");
    }
}

#[test]
#[serial]
fn test_architecture_warnings() {
    // This test primarily serves to emit warnings about testing limitations
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("⚠️  WARNING: Running on non-aarch64 architecture");
        println!("⚠️  SIMD testing is limited - only scalar backend available");
        println!("⚠️  For complete SIMD testing, run on aarch64 (Apple Silicon/ARM64)");
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if !NEON.is_available() {
            println!("⚠️  WARNING: Running on aarch64 but NEON is not available");
            println!("⚠️  This is unusual - NEON should be available on aarch64");
        } else {
            println!("✓ Running on aarch64 with NEON support - full SIMD testing available");
        }
    }
} 