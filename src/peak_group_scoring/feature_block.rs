//! A feature block is a struct of f32 features that is scored as a whole, each field one output column.

/// A block of f32 features.
pub trait FeatureBlock {
    /// Column names, in the order of `values`
    const NAMES: &'static [&'static str];

    fn values(&self) -> Vec<f32>;
}

/// Defines a feature block from its field list: the struct, and `FeatureBlock` with each field's name as its column
/// name. The names and values cannot get out of order, because both follow the one field list.
macro_rules! feature_block {
    (
        $(#[$struct_meta:meta])*
        pub struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                pub $field:ident: f32,
            )*
        }
    ) => {
        $(#[$struct_meta])*
        #[derive(Debug, Clone, Copy, Default, PartialEq)]
        pub struct $name {
            $(
                $(#[$field_meta])*
                pub $field: f32,
            )*
        }

        impl $crate::peak_group_scoring::feature_block::FeatureBlock for $name {
            const NAMES: &'static [&'static str] = &[$(stringify!($field)),*];

            fn values(&self) -> Vec<f32> {
                vec![$(self.$field),*]
            }
        }
    };
}

pub(crate) use feature_block;
