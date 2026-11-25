//! Cross-candidate features: fields of a feature block that compare a candidate with the other candidates of the
//! run, so they can only be computed once every candidate is scored. Example: `ion_fisher_lib` scores a candidate's
//! fragments against an ion-class noise model fitted on all non-best candidates.
//!
//! Every such block follows one pattern:
//! 1. `score_candidate_generic` computes the block's per-candidate fields, leaves the cross-candidate fields at 0,
//!    and puts what the cross-candidate step needs into one field of `CandidateContext`. The field stays empty if
//!    the feature set has no such block.
//! 2. `apply_cross_candidate_features` runs one step per block, in block order: it reads that field of all
//!    candidates, writes the cross-candidate fields of the block, and frees the field before the next step.
//!
//! Context inputs are held for every candidate at once, so they dominate the memory of scoring: keep them as small
//! as possible (indices and a few numbers, never XICs).

use rayon::prelude::*;

use crate::peak_group_scoring::feature_set::{FeatureBlockId, FeatureSet};
use crate::peak_group_scoring::features::CandidateFeature;
use crate::peak_group_scoring::ion_coverage::{num_ion_classes, IonEvidence, IonNoiseModel};

/// What one candidate hands to the cross-candidate step, one field per block. A field is empty if the feature set
/// has no such block.
#[derive(Debug, Default)]
pub struct CandidateContext {
    /// ion-coverage block: the fragment evidence for the noise model of `ion_fisher_lib`
    pub ion_evidence: Vec<IonEvidence>,
}

/// Computes the cross-candidate fields of every block of `feature_set`. `features` and `contexts` are in the same
/// candidate order.
pub fn apply_cross_candidate_features(
    feature_set: FeatureSet,
    top_k_fragments: usize,
    features: &mut [CandidateFeature],
    mut contexts: Vec<CandidateContext>,
) {
    debug_assert_eq!(features.len(), contexts.len());

    if feature_set.has_block(FeatureBlockId::IonCoverage) {
        set_ion_fisher_lib(features, &contexts, top_k_fragments);
        contexts
            .par_iter_mut()
            .for_each(|context| context.ion_evidence = Vec::new());
    }
}

/// The ion-class noise model is fitted on the candidates that are not the best of their precursor, targets and
/// decoys alike, and then scores every candidate's fragments.
fn set_ion_fisher_lib(
    features: &mut [CandidateFeature],
    contexts: &[CandidateContext],
    top_k_fragments: usize,
) {
    let noise_model = IonNoiseModel::fit(
        features
            .par_iter()
            .zip(contexts.par_iter())
            .filter(|(feature, _)| feature.rank >= 1)
            .map(|(_, context)| context.ion_evidence.as_slice()),
        num_ion_classes(top_k_fragments),
    );
    features
        .par_iter_mut()
        .zip(contexts.par_iter())
        .for_each(|(feature, context)| {
            if let Some(ion_coverage) = &mut feature.ion_coverage {
                ion_coverage.ion_fisher_lib = noise_model.fisher(&context.ion_evidence);
            }
        });
}
