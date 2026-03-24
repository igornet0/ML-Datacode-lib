//! Domain handles for the ML plugin — not used by the VM core.
//! Tensors are represented as [`crate::vm_value::Value::Tensor`] (opaque id); other kinds use `Value::PluginOpaque`.

use std::fmt;

use crate::vm_value::Value;

/// Discriminant for [`MlHandle`]. Stable `repr(u8)` for FFI; must match plugin opaque tags.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MlValueKind {
    Tensor = 0,
    Graph = 1,
    LinearRegression = 2,
    Sgd = 3,
    Momentum = 4,
    Nag = 5,
    Adagrad = 6,
    Rmsprop = 7,
    Adam = 8,
    AdamW = 9,
    Dataset = 10,
    NeuralNetwork = 11,
    Sequential = 12,
    Layer = 13,
}

impl TryFrom<u8> for MlValueKind {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Tensor),
            1 => Ok(Self::Graph),
            2 => Ok(Self::LinearRegression),
            3 => Ok(Self::Sgd),
            4 => Ok(Self::Momentum),
            5 => Ok(Self::Nag),
            6 => Ok(Self::Adagrad),
            7 => Ok(Self::Rmsprop),
            8 => Ok(Self::Adam),
            9 => Ok(Self::AdamW),
            10 => Ok(Self::Dataset),
            11 => Ok(Self::NeuralNetwork),
            12 => Ok(Self::Sequential),
            13 => Ok(Self::Layer),
            _ => Err(()),
        }
    }
}

/// Opaque reference to ML runtime state (owned by this dylib).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MlHandle {
    pub kind: MlValueKind,
    pub id: u64,
}

impl MlHandle {
    #[inline]
    pub const fn new(kind: MlValueKind, id: u64) -> Self {
        Self { kind, id }
    }

    #[inline]
    pub const fn kind_u8(self) -> u8 {
        self.kind as u8
    }
}

impl fmt::Debug for MlHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlHandle({:?}, {})", self.kind, self.id)
    }
}

impl fmt::Display for MlHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<ml {:?} id={}>", self.kind, self.id)
    }
}

impl From<MlHandle> for Value {
    fn from(h: MlHandle) -> Self {
        match h.kind {
            MlValueKind::Tensor => Value::Tensor(h.id),
            _ => Value::PluginOpaque {
                tag: h.kind as u8,
                id: h.id,
            },
        }
    }
}
