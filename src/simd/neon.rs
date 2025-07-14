#[cfg(target_arch = "aarch64")]
use super::{SimdBackend, Rank};

#[cfg(target_arch = "aarch64")]
cpufeatures::new!(neon_check, "neon");

#[cfg(target_arch = "aarch64")]
pub struct NeonBackend;

#[cfg(target_arch = "aarch64")]
impl SimdBackend for NeonBackend {
    fn test_backend(&self) -> String {
        // Dummy function to track that neon backend was called
        "aarch64_neon".to_string()
    }
    
    fn name(&self) -> &'static str { 
        "neon" 
    }
    
    fn is_available(&self) -> bool { 
        neon_check::get()
    }
    
    fn priority(&self) -> Rank { 
        Rank::Neon 
    }
} 