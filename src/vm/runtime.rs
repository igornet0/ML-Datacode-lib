//! Opaque object storage for ML runtime handles (`Value::Tensor` and `Value::PluginOpaque`; ids are unique across all kinds).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vm_value::Value;

use crate::ml_types::{MlHandle, MlValueKind};

use crate::dataset::Dataset;
use crate::datasets::DatasetType;
use crate::graph::Graph;
use crate::layer::{LayerId, Sequential};
use crate::model::{LinearRegression, NeuralNetwork};
use crate::optimizer::{Adam, SGD};
use crate::tensor::Tensor;

static NEXT_ML_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static OBJECTS: RefCell<HashMap<u64, MlObject>> = RefCell::new(HashMap::new());
}

pub enum MlObject {
    Tensor(Rc<RefCell<Tensor>>),
    Graph(Rc<RefCell<Graph>>),
    LinearRegression(Rc<RefCell<LinearRegression>>),
    NeuralNetwork(Rc<RefCell<NeuralNetwork>>),
    Sequential(Rc<RefCell<Sequential>>),
    Dataset(Rc<RefCell<Dataset>>),
    /// `load_dataset("mnist"|...)` — materialize on `.split(...)`.
    DatasetCatalog(DatasetType),
    Sgd(Rc<RefCell<SGD>>),
    Adam(Rc<RefCell<Adam>>),
}

fn alloc_id() -> u64 {
    NEXT_ML_ID.fetch_add(1, Ordering::Relaxed)
}

pub fn insert_object(obj: MlObject, kind: MlValueKind) -> MlHandle {
    let id = alloc_id();
    OBJECTS.with(|m| m.borrow_mut().insert(id, obj));
    MlHandle::new(kind, id)
}

pub fn tensor_to_value(t: Tensor) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::Tensor(Rc::new(RefCell::new(t))));
    });
    Value::Tensor(id)
}

pub fn graph_to_value(g: Graph) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::Graph(Rc::new(RefCell::new(g))));
    });
    MlHandle::new(MlValueKind::Graph, id).into()
}

pub fn linear_regression_to_value(model: LinearRegression) -> Value {
    let id = alloc_id();
    OBJECTS.with(|cells| {
        cells.borrow_mut().insert(
            id,
            MlObject::LinearRegression(Rc::new(RefCell::new(model))),
        );
    });
    MlHandle::new(MlValueKind::LinearRegression, id).into()
}

pub fn neural_network_to_value(nn: NeuralNetwork) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut().insert(
            id,
            MlObject::NeuralNetwork(Rc::new(RefCell::new(nn))),
        );
    });
    MlHandle::new(MlValueKind::NeuralNetwork, id).into()
}

pub fn sequential_to_value(s: Sequential) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::Sequential(Rc::new(RefCell::new(s))));
    });
    MlHandle::new(MlValueKind::Sequential, id).into()
}

pub fn dataset_to_value(d: Dataset) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::Dataset(Rc::new(RefCell::new(d))));
    });
    MlHandle::new(MlValueKind::Dataset, id).into()
}

pub fn dataset_catalog_to_value(kind: DatasetType) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::DatasetCatalog(kind));
    });
    MlHandle::new(MlValueKind::DatasetCatalog, id).into()
}

pub fn sgd_to_value(o: SGD) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::Sgd(Rc::new(RefCell::new(o))));
    });
    MlHandle::new(MlValueKind::Sgd, id).into()
}

pub fn adam_to_value(o: Adam) -> Value {
    let id = alloc_id();
    OBJECTS.with(|m| {
        m.borrow_mut()
            .insert(id, MlObject::Adam(Rc::new(RefCell::new(o))));
    });
    MlHandle::new(MlValueKind::Adam, id).into()
}

pub fn layer_value(layer_id: LayerId) -> Value {
    MlHandle::new(MlValueKind::Layer, layer_id as u64).into()
}

pub fn get_tensor(id: u64) -> Option<Rc<RefCell<Tensor>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::Tensor(t) => Some(Rc::clone(t)),
            _ => None,
        })
    })
}

pub fn get_graph(id: u64) -> Option<Rc<RefCell<Graph>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::Graph(g) => Some(Rc::clone(g)),
            _ => None,
        })
    })
}

pub fn get_linear_regression(id: u64) -> Option<Rc<RefCell<LinearRegression>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::LinearRegression(x) => Some(Rc::clone(x)),
            _ => None,
        })
    })
}

pub fn get_neural_network(id: u64) -> Option<Rc<RefCell<NeuralNetwork>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::NeuralNetwork(x) => Some(Rc::clone(x)),
            _ => None,
        })
    })
}

pub fn get_sequential(id: u64) -> Option<Rc<RefCell<Sequential>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::Sequential(x) => Some(Rc::clone(x)),
            _ => None,
        })
    })
}

pub fn get_dataset(id: u64) -> Option<Rc<RefCell<Dataset>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::Dataset(x) => Some(Rc::clone(x)),
            _ => None,
        })
    })
}

pub fn get_dataset_catalog_kind(id: u64) -> Option<DatasetType> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::DatasetCatalog(k) => Some(*k),
            _ => None,
        })
    })
}

pub fn get_sgd(id: u64) -> Option<Rc<RefCell<SGD>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::Sgd(x) => Some(Rc::clone(x)),
            _ => None,
        })
    })
}

pub fn get_adam(id: u64) -> Option<Rc<RefCell<Adam>>> {
    OBJECTS.with(|m| {
        m.borrow().get(&id).and_then(|o| match o {
            MlObject::Adam(x) => Some(Rc::clone(x)),
            _ => None,
        })
    })
}

pub fn as_tensor_ref(v: &Value) -> Option<Rc<RefCell<Tensor>>> {
    match v {
        Value::Tensor(id) => get_tensor(*id),
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Tensor as u8 => get_tensor(*id),
        _ => None,
    }
}

/// Clone tensor payload for `Value::Tensor` or legacy tensor `PluginOpaque`.
#[inline]
pub fn tensor_data_clone(v: &Value) -> Option<Tensor> {
    as_tensor_ref(v).map(|r| r.borrow().clone())
}

/// Build a dense tensor from nested `Value::Array` of numbers (row-major, rectangular).
/// Used when script passes plain arrays instead of `Tensor` handles (e.g. `dataset(features=[...], ...)`).
pub fn tensor_from_nested_numeric_array(v: &Value) -> Option<Tensor> {
    let (flat, shape) = flatten_nested_numeric_array(v)?;
    let expected: usize = shape.iter().product();
    if flat.len() != expected {
        return None;
    }
    Some(Tensor::from_slice(&flat, &shape))
}

/// Prefer existing tensor; otherwise nested numeric arrays.
pub fn tensor_from_value_flexible(v: &Value) -> Option<Tensor> {
    tensor_data_clone(v).or_else(|| tensor_from_nested_numeric_array(v))
}

fn flatten_nested_numeric_array(v: &Value) -> Option<(Vec<f32>, Vec<usize>)> {
    match v {
        Value::Number(n) => Some((vec![*n as f32], vec![1])),
        Value::ByteBuffer(bb) => {
            let slice = &bb.bytes[bb.offset..bb.offset + bb.len];
            let data: Vec<f32> = slice.iter().map(|&b| b as f32).collect();
            Some((data, vec![bb.len]))
        }
        Value::Array(arr) => {
            let arr = arr.borrow();
            if arr.is_empty() {
                return None;
            }
            // Do not branch only on arr[0]: a mix of scalars and row arrays (e.g. first row wrong)
            // would pick the 1D path and fail later with an opaque error.
            let all_numbers = arr.iter().all(|x| matches!(x, Value::Number(_)));
            let all_arrays = arr.iter().all(|x| matches!(x, Value::Array(_)));
            if all_numbers {
                let mut out = Vec::with_capacity(arr.len());
                for x in arr.iter() {
                    if let Value::Number(n) = x {
                        out.push(*n as f32);
                    } else {
                        return None;
                    }
                }
                return Some((out, vec![arr.len()]));
            }
            if all_arrays {
                let mut inner_shape: Option<Vec<usize>> = None;
                let mut flat = Vec::new();
                for row in arr.iter() {
                    let (sub_flat, sub_shape) = flatten_nested_numeric_array(row)?;
                    if let Some(ref prev) = inner_shape {
                        if *prev != sub_shape {
                            return None;
                        }
                    } else {
                        inner_shape = Some(sub_shape);
                    }
                    flat.extend(sub_flat);
                }
                let mut full_shape = vec![arr.len()];
                full_shape.extend(inner_shape?);
                return Some((flat, full_shape));
            }
            let all_tensors = arr.iter().all(|x| tensor_data_clone(x).is_some());
            if all_tensors {
                let mut inner_shape: Option<Vec<usize>> = None;
                let mut flat = Vec::new();
                for row in arr.iter() {
                    let t = tensor_data_clone(row)?;
                    let cpu = t.to_cpu().ok()?;
                    let sub_shape = cpu.shape().to_vec();
                    let sub_flat = cpu.to_vec();
                    if let Some(ref prev) = inner_shape {
                        if *prev != sub_shape {
                            return None;
                        }
                    } else {
                        inner_shape = Some(sub_shape);
                    }
                    flat.extend(sub_flat);
                }
                let mut full_shape = vec![arr.len()];
                full_shape.extend(inner_shape?);
                return Some((flat, full_shape));
            }
            None
        }
        _ => None,
    }
}

/// Detects top-level mix of scalars and row arrays (invalid for `dataset_from_tensors`).
pub fn features_array_mixed_rows_hint(v: &Value) -> Option<&'static str> {
    let Value::Array(rc) = v else {
        return None;
    };
    let b = rc.borrow();
    if b.len() < 2 {
        return None;
    }
    let mut saw_num = false;
    let mut saw_arr = false;
    for x in b.iter() {
        match x {
            Value::Number(_) => saw_num = true,
            Value::Array(_) => saw_arr = true,
            _ => {}
        }
    }
    if saw_num && saw_arr {
        Some(
            "Top-level mix of scalars and row arrays: use a uniform layout [N][feature_dim] (each row an array of numbers), not a mix of ints and arrays.",
        )
    } else {
        None
    }
}

pub fn as_graph_ref(v: &Value) -> Option<Rc<RefCell<Graph>>> {
    match v {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => get_graph(*id),
        _ => None,
    }
}

pub fn as_layer_id(v: &Value) -> Option<LayerId> {
    match v {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Layer as u8 => Some(*id as LayerId),
        _ => None,
    }
}
