//! Train vs eval for dropout / batch norm (thread-local, set by NeuralNetwork::forward).

use std::cell::Cell;

thread_local! {
    static TRAINING: Cell<bool> = Cell::new(false);
}

pub fn set_forward_training(v: bool) {
    TRAINING.with(|c| c.set(v));
}

pub fn forward_training() -> bool {
    TRAINING.with(|c| c.get())
}
