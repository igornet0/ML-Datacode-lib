//! Opaque object storage for ML runtime handles (`Value::Tensor` and `Value::PluginOpaque`; ids are unique across all kinds).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vm_value::Value;

use crate::ml_types::{MlHandle, MlValueKind};

use crate::dataset::Dataset;
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
