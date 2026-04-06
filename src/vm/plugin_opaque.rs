//! Numeric tags and display names for `Value::PluginOpaque` — used only inside the ML dylib.

pub const TAG_TENSOR: u8 = 0;
pub const TAG_GRAPH: u8 = 1;
pub const TAG_LINEAR_REGRESSION: u8 = 2;
pub const TAG_SGD: u8 = 3;
pub const TAG_MOMENTUM: u8 = 4;
pub const TAG_NAG: u8 = 5;
pub const TAG_ADAGRAD: u8 = 6;
pub const TAG_RMSPROP: u8 = 7;
pub const TAG_ADAM: u8 = 8;
pub const TAG_ADAMW: u8 = 9;
pub const TAG_DATASET: u8 = 10;
pub const TAG_NEURAL_NETWORK: u8 = 11;
pub const TAG_SEQUENTIAL: u8 = 12;
pub const TAG_LAYER: u8 = 13;
pub const TAG_BOUND_METHOD: u8 = 14;
pub const TAG_DATASET_CATALOG: u8 = 15;

pub fn plugin_opaque_builtin_type_name(tag: u8) -> &'static str {
    match tag {
        TAG_TENSOR => "tensor",
        TAG_GRAPH => "graph",
        TAG_LINEAR_REGRESSION => "linear_regression",
        TAG_SGD => "sgd",
        TAG_MOMENTUM => "momentum",
        TAG_NAG => "nag",
        TAG_ADAGRAD => "adagrad",
        TAG_RMSPROP => "rmsprop",
        TAG_ADAM => "adam",
        TAG_ADAMW => "adamw",
        TAG_DATASET => "dataset",
        TAG_NEURAL_NETWORK => "neural_network",
        TAG_SEQUENTIAL => "sequential",
        TAG_LAYER => "layer",
        TAG_BOUND_METHOD => "bound_method",
        TAG_DATASET_CATALOG => "dataset_catalog",
        _ => "plugin_opaque",
    }
}

pub fn plugin_opaque_schema_label(tag: u8) -> &'static str {
    match tag {
        TAG_TENSOR => "Tensor",
        TAG_GRAPH => "Graph",
        TAG_LINEAR_REGRESSION => "LinearRegression",
        TAG_SGD => "SGD",
        TAG_MOMENTUM => "Momentum",
        TAG_NAG => "NAG",
        TAG_ADAGRAD => "Adagrad",
        TAG_RMSPROP => "RMSprop",
        TAG_ADAM => "Adam",
        TAG_ADAMW => "AdamW",
        TAG_DATASET => "Dataset",
        TAG_NEURAL_NETWORK => "NeuralNetwork",
        TAG_SEQUENTIAL => "Sequential",
        TAG_LAYER => "Layer",
        TAG_BOUND_METHOD => "BoundMethod",
        TAG_DATASET_CATALOG => "DatasetCatalog",
        _ => "PluginOpaque",
    }
}
