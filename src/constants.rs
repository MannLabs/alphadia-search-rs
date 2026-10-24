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

/// Width in Th of the logistic edges of the quadrupole transmission. Fitted on the fragment signal of
/// precursors split across two windows (200 ng HeLa, Astral, 2 Th windows): 0.08 Th.
pub const QUADRUPOLE_EDGE_WIDTH: f32 = 0.08;

/// Highest fragment isotope considered. Each fragment is extracted at its most intense isotope in the
/// window: in a window that holds M1 and M2 of a precursor, long fragments carry most of their signal at
/// their first or second isotope.
pub const MAX_FRAGMENT_ISOTOPE: usize = 2;

/// Windows whose share of the isotope envelope is within this fraction of the largest share are all
/// primary windows (overlapping windows).
pub const PRIMARY_WINDOW_TOLERANCE: f32 = 0.05;

/// A secondary window must hold at least this share of the primary window's signal. Below it, its fragments
/// are mostly below the detection limit and their zeros carry no information.
pub const SECONDARY_MIN_SIGNAL_RATIO: f32 = 0.25;
