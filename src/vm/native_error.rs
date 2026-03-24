//! Error reporting from ML natives (host may wire this to websocket in the future).

pub fn set_native_error(msg: String) {
    eprintln!("[ml] {}", msg);
}
