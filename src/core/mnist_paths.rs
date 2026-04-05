//! Относительные пути к MNIST IDX от корня нативного пакета (`packages/<name>/` в DPM).

pub const TRAIN_IMAGES: &str = "datasets/mnist/train-images.idx3-ubyte";
pub const TRAIN_LABELS: &str = "datasets/mnist/train-labels.idx1-ubyte";
pub const T10K_IMAGES: &str = "datasets/mnist/t10k-images.idx3-ubyte";
pub const T10K_LABELS: &str = "datasets/mnist/t10k-labels.idx1-ubyte";

pub fn relative_for_split(split: &str) -> Option<(&'static str, &'static str)> {
    match split {
        "train" => Some((TRAIN_IMAGES, TRAIN_LABELS)),
        "test" => Some((T10K_IMAGES, T10K_LABELS)),
        _ => None,
    }
}
