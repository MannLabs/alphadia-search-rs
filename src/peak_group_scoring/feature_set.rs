//! Which features `PeakGroupScoring` computes: the calibration rounds score the core block only, the final round all
//! blocks. The window-share, MS1 and detection blocks come last, so that the columns of the other blocks keep
//! their positions.

use pyo3::prelude::*;

use crate::peak_group_scoring::core::CoreFeatures;
use crate::peak_group_scoring::detection::DetectionFeatures;
use crate::peak_group_scoring::feature_block::FeatureBlock;
use crate::peak_group_scoring::fragment_isotope::FragmentIsotopeFeatures;
use crate::peak_group_scoring::ion_coverage::IonCoverageFeatures;
use crate::peak_group_scoring::library_agreement::LibraryAgreementFeatures;
use crate::peak_group_scoring::ms1::Ms1Features;
use crate::peak_group_scoring::window_share::WindowShareFeatures;

#[pyclass(eq, eq_int, hash, frozen, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    FragmentIsotope,
    WindowShare,
    Ms1,
    Detection,
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
                FeatureBlockId::FragmentIsotope,
                FeatureBlockId::WindowShare,
                FeatureBlockId::Ms1,
                FeatureBlockId::Detection,
            ],
        }
    }

    /// Whether the M+1 XIC of each isotope-eligible fragment adds to its fragment XIC. The calibration rounds use
    /// the monoisotopic XICs only.
    pub fn sums_fragment_isotopes(self) -> bool {
        self == FeatureSet::Full
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
            FeatureBlockId::FragmentIsotope => FragmentIsotopeFeatures::NAMES,
            FeatureBlockId::WindowShare => WindowShareFeatures::NAMES,
            FeatureBlockId::Ms1 => Ms1Features::NAMES,
            FeatureBlockId::Detection => DetectionFeatures::NAMES,
        }
    }
}
