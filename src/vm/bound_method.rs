//! Registry for `PluginOpaque` bound methods (NN and dataset callables).

use std::cell::RefCell;

use crate::ml_types::MlValueKind;
use crate::vm_value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnBoundMethodKind {
    Device,
    GetDevice,
    Train,
    Save,
}

/// Payload for [`MlValueKind::BoundMethod`] thunks (NN methods or `dataset.split`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundMethodPayload {
    Nn {
        nn_id: u64,
        kind: NnBoundMethodKind,
    },
    DatasetSplit {
        dataset_id: u64,
    },
    DatasetConcat {
        dataset_id: u64,
    },
    DatasetPushData {
        dataset_id: u64,
    },
}

thread_local! {
    static BOUND_REGISTRY: RefCell<Vec<BoundMethodPayload>> = RefCell::new(Vec::new());
}

fn register_payload(payload: BoundMethodPayload) -> u64 {
    BOUND_REGISTRY.with(|r| {
        let mut v = r.borrow_mut();
        let id = v.len() as u64;
        v.push(payload);
        id
    })
}

pub fn lookup_bound(id: u64) -> Option<BoundMethodPayload> {
    BOUND_REGISTRY.with(|r| r.borrow().get(id as usize).copied())
}

pub fn register_nn_bound_method(nn_id: u64, kind: NnBoundMethodKind) -> u64 {
    register_payload(BoundMethodPayload::Nn { nn_id, kind })
}

pub fn lookup_nn_bound(id: u64) -> Option<(u64, NnBoundMethodKind)> {
    match lookup_bound(id)? {
        BoundMethodPayload::Nn { nn_id, kind } => Some((nn_id, kind)),
        _ => None,
    }
}

pub fn register_dataset_split_bound(dataset_id: u64) -> u64 {
    register_payload(BoundMethodPayload::DatasetSplit { dataset_id })
}

pub fn register_dataset_concat_bound(dataset_id: u64) -> u64 {
    register_payload(BoundMethodPayload::DatasetConcat { dataset_id })
}

pub fn register_dataset_push_data_bound(dataset_id: u64) -> u64 {
    register_payload(BoundMethodPayload::DatasetPushData { dataset_id })
}

pub fn nn_method_kind_for_name(name: &str) -> Option<NnBoundMethodKind> {
    match name {
        "device" => Some(NnBoundMethodKind::Device),
        "get_device" => Some(NnBoundMethodKind::GetDevice),
        "train" => Some(NnBoundMethodKind::Train),
        "save" => Some(NnBoundMethodKind::Save),
        _ => None,
    }
}

pub fn bound_method_value(nn_id: u64, kind: NnBoundMethodKind) -> Value {
    let id = register_nn_bound_method(nn_id, kind);
    Value::PluginOpaque {
        tag: MlValueKind::BoundMethod as u8,
        id,
    }
}

pub fn bound_dataset_split_value(dataset_id: u64) -> Value {
    let id = register_dataset_split_bound(dataset_id);
    Value::PluginOpaque {
        tag: MlValueKind::BoundMethod as u8,
        id,
    }
}

pub fn bound_dataset_concat_value(dataset_id: u64) -> Value {
    let id = register_dataset_concat_bound(dataset_id);
    Value::PluginOpaque {
        tag: MlValueKind::BoundMethod as u8,
        id,
    }
}

pub fn bound_dataset_push_data_value(dataset_id: u64) -> Value {
    let id = register_dataset_push_data_bound(dataset_id);
    Value::PluginOpaque {
        tag: MlValueKind::BoundMethod as u8,
        id,
    }
}
