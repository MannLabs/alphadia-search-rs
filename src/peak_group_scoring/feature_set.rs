//! Which features `PeakGroupScoring` computes. A feature set is a list of feature blocks; the split follows the FDR
//! rounds of alphaDIA (MannLabs/alphadia#910).

use pyo3::prelude::*;

use crate::peak_group_scoring::core::CoreFeatures;
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::ion_coverage::IonCoverageFeatures;
use crate::peak_group_scoring::library_agreement::LibraryAgreementFeatures;

/// The feature set of a scoring run.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSet {
    Full,
    Calibration,
}

/// A feature block, each a struct of f32 features in its own module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    Core,
    IonCoverage,
    LibraryAgreement,
}

impl FeatureSet {
    /// The blocks of each feature set, in column order.
    pub fn blocks(self) -> &'static [Block] {
        match self {
            FeatureSet::Calibration => &[Block::Core],
            FeatureSet::Full => &[Block::Core, Block::IonCoverage, Block::LibraryAgreement],
        }
    }

    pub fn has(self, block: Block) -> bool {
        self.blocks().contains(&block)
    }

    pub fn feature_names(self) -> Vec<&'static str> {
        self.blocks()
            .iter()
            .flat_map(|block| block.names().iter().copied())
            .collect()
    }
}

impl Block {
    pub fn names(self) -> &'static [&'static str] {
        match self {
            Block::Core => CoreFeatures::NAMES,
            Block::IonCoverage => IonCoverageFeatures::NAMES,
            Block::LibraryAgreement => LibraryAgreementFeatures::NAMES,
        }
    }
}
