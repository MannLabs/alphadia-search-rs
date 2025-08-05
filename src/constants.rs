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
