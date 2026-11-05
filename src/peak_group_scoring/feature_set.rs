//! Which features `PeakGroupScoring` computes: the calibration rounds score the core block only, the final round all
//! blocks.

use pyo3::prelude::*;

use crate::peak_group_scoring::core::CoreFeatures;
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::ion_coverage::IonCoverageFeatures;
use crate::peak_group_scoring::library_agreement::LibraryAgreementFeatures;

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSet {
    Full,
    Calibration,
}

/// Identifies one `FeatureBlock` struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureBlockId {
    Core,
    IonCoverage,
    LibraryAgreement,
}

impl FeatureSet {
    /// The blocks of each feature set, in column order.
    pub fn blocks(self) -> &'static [FeatureBlockId] {
        match self {
            FeatureSet::Calibration => &[FeatureBlockId::Core],
            FeatureSet::Full => &[
                FeatureBlockId::Core,
                FeatureBlockId::IonCoverage,
                FeatureBlockId::LibraryAgreement,
            ],
        }
    }

    pub fn has_block(self, block: FeatureBlockId) -> bool {
        self.blocks().contains(&block)
    }

    pub fn feature_names(self) -> Vec<&'static str> {
        self.blocks()
            .iter()
            .flat_map(|block| block.names().iter().copied())
            .collect()
    }
}

impl FeatureBlockId {
    pub fn names(self) -> &'static [&'static str] {
        match self {
            FeatureBlockId::Core => CoreFeatures::NAMES,
            FeatureBlockId::IonCoverage => IonCoverageFeatures::NAMES,
            FeatureBlockId::LibraryAgreement => LibraryAgreementFeatures::NAMES,
        }
    }
}
