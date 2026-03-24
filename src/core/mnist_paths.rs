//! Bundled MNIST IDX paths relative to this crate (`CARGO_MANIFEST_DIR`).
pub const TRAIN_IMAGES: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/mnist/train-images.idx3-ubyte");
pub const TRAIN_LABELS: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/mnist/train-labels.idx1-ubyte");
pub const T10K_IMAGES: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/mnist/t10k-images.idx3-ubyte");
pub const T10K_LABELS: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/mnist/t10k-labels.idx1-ubyte");
