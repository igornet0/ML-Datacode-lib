//! Value type used inside the ML dylib (no dependency on `data-code`).
//! Mirrors the ABI-representable subset of the VM `Value` enum plus `ObjectPtr` for foreign module receivers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use datacode_abi::NativeHandle;

/// Dense bytes from VM (`read_file_bin`); crosses ABI as `AbiValue::Bytes`.
#[derive(Clone, Debug)]
pub struct ByteBuffer {
    pub bytes: Rc<Vec<u8>>,
    pub offset: usize,
    pub len: usize,
}

impl ByteBuffer {
    pub fn from_vec(v: Vec<u8>) -> Self {
        let len = v.len();
        Self {
            bytes: Rc::new(v),
            offset: 0,
            len,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(String),
    Array(Rc<RefCell<Vec<Value>>>),
    Null,
    ByteBuffer(ByteBuffer),
    /// Tensor stored in ML runtime (`OBJECTS`) by opaque id; ABI round-trip uses `PluginOpaque` with `MlValueKind::Tensor`.
    Tensor(u64),
    PluginOpaque { tag: u8, id: u64 },
    Object(Rc<RefCell<HashMap<String, Value>>>),
    /// Pointer from VM `AbiValue::Object` when it is not a round-tripped local `Object`.
    ObjectPtr(NativeHandle),
    Path(PathBuf),
}
