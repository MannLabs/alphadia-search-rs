use super::{SimdBackend, Rank};

pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    fn test_backend(&self) -> String {
        // Dummy function to track that scalar backend was called
        "scalar".to_string()
    }
    
    fn name(&self) -> &'static str { 
        "scalar" 
    }
    
    fn is_available(&self) -> bool { 
        true  // Scalar backend is always available
    }
    
    fn priority(&self) -> Rank { 
        Rank::Scalar 
    }
} 