use datacode_sdk::dist_rel_path;

pub const CIFAR10_TRAIN_BATCHES: [&str; 5] = [
    dist_rel_path!("/datasets/cifar-10/train_batch_1.bin"),
    dist_rel_path!("/datasets/cifar-10/train_batch_2.bin"),
    dist_rel_path!("/datasets/cifar-10/train_batch_3.bin"),
    dist_rel_path!("/datasets/cifar-10/train_batch_4.bin"),
    dist_rel_path!("/datasets/cifar-10/train_batch_5.bin"),
];
pub const CIFAR10_TEST_BATCH: &str = dist_rel_path!("/datasets/cifar-10/test_batch.bin");

pub const CIFAR100_TRAIN: &str = dist_rel_path!("/datasets/cifar-100/train.bin");
pub const CIFAR100_TEST: &str = dist_rel_path!("/datasets/cifar-100/test.bin");

pub fn cifar10_relative_for_split(split: &str) -> Option<&'static [&'static str]> {
    match split {
        "train" => Some(&CIFAR10_TRAIN_BATCHES),
        "test" => Some(&[CIFAR10_TEST_BATCH]),
        _ => None,
    }
}

pub fn cifar100_relative_for_split(split: &str) -> Option<&'static str> {
    match split {
        "train" => Some(CIFAR100_TRAIN),
        "test" => Some(CIFAR100_TEST),
        _ => None,
    }
}
