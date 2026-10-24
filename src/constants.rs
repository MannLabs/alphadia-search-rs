pub struct FragmentType;

impl FragmentType {
    pub const A: u8 = 97; // ascii value for 'a'
    pub const B: u8 = 98; // ascii value for 'b'
    pub const C: u8 = 99; // ascii value for 'c'
    pub const X: u8 = 120; // ascii value for 'x'
    pub const Y: u8 = 121; // ascii value for 'y'
    pub const Z: u8 = 122; // ascii value for 'z'
}

pub struct Loss;

impl Loss {
    pub const MODLOSS: u8 = 98; // similar to molecular weight of phosphate group
    pub const H2O: u8 = 18; // similar to molecular weight of water molecule
    pub const NH3: u8 = 17; // similar to molecular weight of ammonia molecule
    pub const LOSSH: u8 = 1; // similar to molecular weight of hydrogen atom
    pub const ADDH: u8 = 2; // there is no -1 so we use 2
    pub const NONE: u8 = 0;
}

/// Mass of a proton in Da.
pub const PROTON_MASS: f32 = 1.007_276_5;

/// Mass difference between 13C and 12C in Da, the spacing of peptide isotope peaks.
pub const ISOTOPE_SPACING: f32 = 1.003_354_8;

/// Width in Th of the logistic edges of the quadrupole transmission.
/// Not calibrated: at 0.1 Th the transmission falls from 90 % to 10 % over 0.44 Th around each window edge.
pub const QUADRUPOLE_EDGE_WIDTH: f32 = 0.1;

/// Highest fragment isotope extracted. Fragments of a heavy precursor isotope carry its extra neutrons in part,
/// but extracting the first fragment isotope too gave decoys more chance matches than targets gained signal
/// (local A/B, 200 ng Astral: -11 % targets at 1 % q).
pub const MAX_FRAGMENT_ISOTOPE: usize = 0;

/// Extraction channels expected to carry less than this share of a fragment's signal are skipped:
/// they add little signal but cost an extraction.
pub const MIN_CHANNEL_GAIN: f32 = 0.1;
