use datacode_sdk::dist_rel_path;

pub const TRAIN_IMAGES: &str = dist_rel_path!("/datasets/mnist/train-images.idx3-ubyte");
pub const TRAIN_LABELS: &str = dist_rel_path!("/datasets/mnist/train-labels.idx1-ubyte");
pub const T10K_IMAGES: &str = dist_rel_path!("/datasets/mnist/t10k-images.idx3-ubyte");
pub const T10K_LABELS: &str = dist_rel_path!("/datasets/mnist/t10k-labels.idx1-ubyte");

pub fn mnist_relative_for_split(split: &str) -> Option<(&'static str, &'static str)> {
    match split {
        "train" => Some((TRAIN_IMAGES, TRAIN_LABELS)),
        "test" => Some((T10K_IMAGES, T10K_LABELS)),
        _ => None,
    }
}