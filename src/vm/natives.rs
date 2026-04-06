// Native functions for ML module

use crate::vm_value::Value;
use crate::ml_types::MlValueKind;
use crate::tensor::Tensor;
use crate::graph::{Graph, OpType};
use crate::model::{LinearRegression, NeuralNetwork};
use crate::optimizer::{SGD, Adam};
use crate::loss::{mse_loss, binary_cross_entropy_loss,
                      mae_loss, huber_loss, hinge_loss, kl_divergence, smooth_l1_loss,
                      categorical_cross_entropy_loss};
use crate::dataset::Dataset;
use crate::device::Device;
use std::rc::Rc;
use std::cell::RefCell;
use std::path::PathBuf;

// Thread-local storage for global ML device state
thread_local! {
    static GLOBAL_ML_DEVICE: RefCell<Option<Device>> = RefCell::new(None);
}

/// Строка для встроенного `typeof()` в VM: экспорт `ml.opaque_type_name` вызывает эту функцию по ABI.
/// Имена совпадают с вариантами [`MlValueKind`].
pub fn native_ml_opaque_type_name(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }
    let tag = match &args[0] {
        Value::PluginOpaque { tag, .. } => *tag,
        Value::Tensor(_) => MlValueKind::Tensor as u8,
        _ => return Value::String("plugin_opaque".to_string()),
    };
    let name: &'static str = match MlValueKind::try_from(tag) {
        Ok(MlValueKind::Tensor) => "tensor",
        Ok(MlValueKind::Graph) => "graph",
        Ok(MlValueKind::LinearRegression) => "linear_regression",
        Ok(MlValueKind::Sgd) => "sgd",
        Ok(MlValueKind::Momentum) => "momentum",
        Ok(MlValueKind::Nag) => "nag",
        Ok(MlValueKind::Adagrad) => "adagrad",
        Ok(MlValueKind::Rmsprop) => "rmsprop",
        Ok(MlValueKind::Adam) => "adam",
        Ok(MlValueKind::AdamW) => "adamw",
        Ok(MlValueKind::Dataset) => "dataset",
        Ok(MlValueKind::DatasetCatalog) => "dataset_catalog",
        Ok(MlValueKind::NeuralNetwork) => "neural_network",
        Ok(MlValueKind::Sequential) => "sequential",
        Ok(MlValueKind::Layer) => "layer",
        Ok(MlValueKind::BoundMethod) => "bound_method",
        Err(()) => "plugin_opaque",
    };
    Value::String(name.to_string())
}

/// Короткая строка для вывода (`print`): `<tensor tag=0 id=4>` и т.д. Экспорт `ml.opaque_display`.
pub fn native_ml_opaque_display(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }
    let (tag, id) = match &args[0] {
        Value::PluginOpaque { tag, id } => (*tag, *id),
        Value::Tensor(tid) => (MlValueKind::Tensor as u8, *tid),
        _ => return Value::Null,
    };
    let name: &'static str = match MlValueKind::try_from(tag) {
        Ok(MlValueKind::Tensor) => "tensor",
        Ok(MlValueKind::Graph) => "graph",
        Ok(MlValueKind::LinearRegression) => "linear_regression",
        Ok(MlValueKind::Sgd) => "sgd",
        Ok(MlValueKind::Momentum) => "momentum",
        Ok(MlValueKind::Nag) => "nag",
        Ok(MlValueKind::Adagrad) => "adagrad",
        Ok(MlValueKind::Rmsprop) => "rmsprop",
        Ok(MlValueKind::Adam) => "adam",
        Ok(MlValueKind::AdamW) => "adamw",
        Ok(MlValueKind::Dataset) => "dataset",
        Ok(MlValueKind::DatasetCatalog) => "dataset_catalog",
        Ok(MlValueKind::NeuralNetwork) => "neural_network",
        Ok(MlValueKind::Sequential) => "sequential",
        Ok(MlValueKind::Layer) => "layer",
        Ok(MlValueKind::BoundMethod) => "bound_method",
        Err(()) => "plugin_opaque",
    };
    Value::String(format!("<{} tag={} id={}>", name, tag, id))
}

/// Helper function to recursively extract data and infer shape from nested arrays.
/// `Value::ByteBuffer` rows (e.g. CIFAR pixel rows after `chunk`) flatten to `f32` without per-byte
/// `Value::Number`; the host VM keeps byte buffers as ABI `Bytes` when crossing the dylib boundary.
/// Returns Ok((data, shape)) on success, or Err(error_message) on error
fn extract_tensor_data_and_shape(value: &Value) -> Result<(Vec<f32>, Vec<usize>), String> {
    // Single tensor handle (e.g. row from `one_hot(...)[0]`) — same layout as a dense numeric row
    if let Some(t) = crate::runtime::tensor_data_clone(value) {
        let cpu = t.to_cpu()?;
        let shape = cpu.shape().to_vec();
        let data = cpu.to_vec();
        return Ok((data, shape));
    }
    match value {
        Value::ByteBuffer(bb) => {
            let slice = &bb.bytes[bb.offset..bb.offset + bb.len];
            let data: Vec<f32> = slice.iter().map(|&b| b as f32).collect();
            Ok((data, vec![bb.len]))
        }
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            if arr_ref.is_empty() {
                return Err("Empty array cannot be converted to tensor".to_string());
            }
            
            // Check if all elements are numbers (1D array)
            let mut is_flat = true;
            for val in arr_ref.iter() {
                if !matches!(val, Value::Number(_)) {
                    is_flat = false;
                    break;
                }
            }
            
            if is_flat {
                // Flat array of numbers
                let data: Vec<f32> = arr_ref.iter()
                    .map(|v| {
                        match v {
                            Value::Number(n) => *n as f32,
                            _ => 0.0, // Shouldn't happen due to check above
                        }
                    })
                    .collect();
                let data_len = data.len();
                return Ok((data, vec![data_len]));
            }
            
            // Nested array — one flat buffer; reserve after first row (same shapes required).
            let mut flat_data: Vec<f32> = Vec::new();
            let mut first_shape: Option<Vec<usize>> = None;

            for (idx, val) in arr_ref.iter().enumerate() {
                match extract_tensor_data_and_shape(val) {
                    Ok((data, shape)) => {
                        if let Some(fs) = &first_shape {
                            if shape != *fs {
                                let expected_row_length = fs.iter().product::<usize>();
                                let actual_row_length = shape.iter().product::<usize>();
                                return Err(format!(
                                    "ShapeError: expected row length {}, but got {} at row index {}",
                                    expected_row_length, actual_row_length, idx
                                ));
                            }
                        } else {
                            first_shape = Some(shape);
                            flat_data.reserve(arr_ref.len().saturating_mul(data.len()));
                        }
                        flat_data.extend(data);
                    }
                    Err(msg) => {
                        return Err(format!("Error at row index {}: {}", idx, msg));
                    }
                }
            }

            let Some(fs) = first_shape else {
                return Err("Empty nested array cannot be converted to tensor".to_string());
            };

            let mut tensor_shape = vec![arr_ref.len()];
            tensor_shape.extend(fs);

            Ok((flat_data, tensor_shape))
        }
        Value::Number(n) => {
            // Scalar value
            Ok((vec![*n as f32], vec![1]))
        }
        _ => Err("Invalid value type for tensor creation".to_string()),
    }
}

/// Create a tensor from data array and shape
/// tensor([1.0, 2.0, 3.0]) - auto-infer shape as [3]
/// tensor([[1], [2], [3]]) - auto-infer shape as [3, 1]
/// tensor([1.0, 2.0, 3.0], [3]) - explicit shape
pub fn native_tensor(args: &[Value]) -> Value {
    if args.is_empty() || args.len() > 2 {
        return Value::Null;
    }

    // If only 1 argument, auto-infer shape from nested array structure
    if args.len() == 1 {
        match extract_tensor_data_and_shape(&args[0]) {
            Ok((data, shape)) => {
                match Tensor::new(data, shape) {
                    Ok(tensor) => crate::runtime::tensor_to_value(tensor),
                    Err(err_msg) => {
                        // Set error message for VM to handle
                        use crate::native_error::set_native_error;
                        set_native_error(err_msg);
                        Value::Null
                    }
                }
            }
            Err(err_msg) => {
                // Set error message for VM to handle
                use crate::native_error::set_native_error;
                set_native_error(err_msg);
                Value::Null
            }
        }
    } else {
        // If 2 arguments, extract flat data and use explicit shape
        let data_array = match &args[0] {
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                let mut data = Vec::new();
                for val in arr_ref.iter() {
                    match val {
                        Value::Number(n) => data.push(*n as f32),
                        _ => return Value::Null,
                    }
                }
                data
            }
            _ => return Value::Null,
        };

        // Extract shape array
        let shape_array = match &args[1] {
            Value::Array(arr) => {
                let arr_ref = arr.borrow();
                let mut shape = Vec::new();
                for val in arr_ref.iter() {
                    match val {
                        Value::Number(n) => {
                            let size = *n as i64;
                            if size < 0 {
                                return Value::Null;
                            }
                            shape.push(size as usize);
                        }
                        _ => return Value::Null,
                    }
                }
                shape
            }
            _ => return Value::Null,
        };

        match Tensor::new(data_array, shape_array) {
            Ok(tensor) => crate::runtime::tensor_to_value(tensor),
            Err(_) => Value::Null,
        }
    }
}

/// Get tensor shape
/// shape(tensor)
pub fn native_shape(args: &[Value]) -> Value {
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error(format!("ml.shape expects 1 argument (tensor), got {}", args.len()));
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        use crate::native_error::set_native_error;
        set_native_error("ml.shape: argument must be a Tensor".to_string());
        return Value::Null;
    };

    let shape = tensor.borrow().shape.clone();
    let shape_values: Vec<Value> = shape.iter().map(|&s| Value::Number(s as f64)).collect();
    Value::Array(Rc::new(RefCell::new(shape_values)))
}

/// Global maximum of all tensor elements (for `max(tensor)` via VM `native_plugin_call`).
pub fn native_tensor_max(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }
    let Some(t) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };
    let guard = t.borrow();
    let data = guard.as_slice();
    if data.is_empty() {
        return Value::Null;
    }
    let m = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Value::Number(f32_to_display_f64(m))
}

/// Global minimum of all tensor elements (for `min(tensor)` via VM `native_plugin_call`).
pub fn native_tensor_min(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }
    let Some(t) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };
    let guard = t.borrow();
    let data = guard.as_slice();
    if data.is_empty() {
        return Value::Null;
    }
    let m = data.iter().copied().fold(f32::INFINITY, f32::min);
    Value::Number(f32_to_display_f64(m))
}

/// Get tensor data as array
/// data(tensor)
pub fn native_data(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let data = tensor.borrow().to_vec();
    let data_values: Vec<Value> = data.iter().map(|&d| Value::Number(d as f64)).collect();
    Value::Array(Rc::new(RefCell::new(data_values)))
}

/// Округление f32→f64 для печати: убирает шум IEEE754 (например 0.89999998 вместо 0.9).
fn f32_to_display_f64(x: f32) -> f64 {
    let d = f64::from(x);
    (d * 1_000_000.0).round() / 1_000_000.0
}

fn format_f32_repr(x: f32) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    // Достаточно ~7 значащих цифр для f32; {:.6} убирает артефакты вроде 0.89999998
    let s = format!("{:.6}", x);
    s.trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn format_nested_inner(shape: &[usize], data: &[f32], pos: &mut usize) -> Result<String, ()> {
    match shape.len() {
        0 => {
            if *pos >= data.len() {
                return Err(());
            }
            let s = format_f32_repr(data[*pos]);
            *pos += 1;
            Ok(s)
        }
        1 => {
            let k = shape[0];
            let mut parts = Vec::with_capacity(k);
            for _ in 0..k {
                if *pos >= data.len() {
                    return Err(());
                }
                parts.push(format_f32_repr(data[*pos]));
                *pos += 1;
            }
            Ok(format!("[{}]", parts.join(", ")))
        }
        _ => {
            let k = shape[0];
            let mut rows = Vec::with_capacity(k);
            for _ in 0..k {
                rows.push(format_nested_inner(&shape[1..], data, pos)?);
            }
            Ok(format!(
                "[\n{}\n]",
                rows
                    .iter()
                    .map(|r| format!("  {}", r))
                    .collect::<Vec<_>>()
                    .join(",\n")
            ))
        }
    }
}

/// Строковое представление тензора по `shape` (row-major). Экспорт через `native_plugin_call(_, "repr")`.
pub fn native_tensor_repr(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }
    let Some(t) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };
    let tb = t.borrow();
    let shape = &tb.shape;
    let data = tb.as_slice();
    if data.is_empty() {
        return Value::String("[]".to_string());
    }
    let expected: usize = shape.iter().product();
    if !shape.is_empty() && expected != data.len() {
        return Value::Null;
    }
    const MAX_ELEMS: usize = 4096;
    if data.len() > MAX_ELEMS {
        let parts: Vec<String> = data[..MAX_ELEMS]
            .iter()
            .map(|x| format_f32_repr(*x))
            .collect();
        return Value::String(format!(
            "[{}] ... (truncated, {} of {} elements)",
            parts.join(", "),
            MAX_ELEMS,
            data.len()
        ));
    }
    let mut pos = 0usize;
    match format_nested_inner(shape, data, &mut pos) {
        Ok(s) => Value::String(s),
        Err(()) => Value::Null,
    }
}

/// Element-wise addition
/// add(t1, t2)
pub fn native_add(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(t1) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let Some(t2) = crate::runtime::as_tensor_ref(&args[1]) else {
        return Value::Null;
    };

    let a = t1.borrow().clone();
    let b = t2.borrow().clone();
    match a.add(&b) {
        Ok(result) => crate::runtime::tensor_to_value(result),
        Err(_) => Value::Null,
    }
}

/// Element-wise subtraction
/// sub(t1, t2)
pub fn native_sub(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(t1) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let Some(t2) = crate::runtime::as_tensor_ref(&args[1]) else {
        return Value::Null;
    };

    let a = t1.borrow().clone();
    let b = t2.borrow().clone();
    match a.sub(&b) {
        Ok(result) => crate::runtime::tensor_to_value(result),
        Err(_) => Value::Null,
    }
}

/// Element-wise multiplication
/// mul(t1, t2)
pub fn native_mul(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(t1) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let Some(t2) = crate::runtime::as_tensor_ref(&args[1]) else {
        return Value::Null;
    };

    let a = t1.borrow().clone();
    let b = t2.borrow().clone();
    match a.mul(&b) {
        Ok(result) => crate::runtime::tensor_to_value(result),
        Err(_) => Value::Null,
    }
}

/// Matrix multiplication
/// matmul(t1, t2)
pub fn native_matmul(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(t1) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let Some(t2) = crate::runtime::as_tensor_ref(&args[1]) else {
        return Value::Null;
    };

    let a = t1.borrow().clone();
    let b = t2.borrow().clone();
    match a.matmul(&b) {
        Ok(result) => crate::runtime::tensor_to_value(result),
        Err(_) => Value::Null,
    }
}

/// Operator registration for the host parser: one row per symbol — `@` → `matmul` at factor precedence (60), left-assoc.
pub fn native_operator_descriptor(_args: &[Value]) -> Value {
    use std::cell::RefCell;
    use std::rc::Rc;
    Value::Array(Rc::new(RefCell::new(vec![Value::Array(Rc::new(RefCell::new(vec![
        Value::String("@".to_string()),
        Value::String("matmul".to_string()),
        Value::Number(60.0),
        Value::Number(0.0),
    ])))])))
}

/// Host compiler parse-time kwargs: rows `[export_key, param1, ...]` and `["__method__", method, export_key]`
/// for ambiguous builtin vs plugin methods (see [`crate::dataset_split_abi::DATASET_SPLIT_NAMED_ARG_NAMES`]).
pub fn native_call_descriptor(_args: &[Value]) -> Value {
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut row_split = Vec::with_capacity(1 + crate::dataset_split_abi::DATASET_SPLIT_NAMED_ARG_NAMES.len());
    row_split.push(Value::String("native_dataset_split".to_string()));
    row_split.extend(
        crate::dataset_split_abi::DATASET_SPLIT_NAMED_ARG_NAMES
            .iter()
            .map(|s| Value::String((*s).to_string())),
    );
    let mut row_push = Vec::with_capacity(1 + crate::dataset_push_data_abi::DATASET_PUSH_DATA_NAMED_ARG_NAMES.len());
    row_push.push(Value::String("native_dataset_push_data".to_string()));
    row_push.extend(
        crate::dataset_push_data_abi::DATASET_PUSH_DATA_NAMED_ARG_NAMES
            .iter()
            .map(|s| Value::String((*s).to_string())),
    );
    let row_concat = vec![
        Value::String("native_dataset_concat".to_string()),
        Value::String("other".to_string()),
    ];
    let method_split = vec![
        Value::String("__method__".to_string()),
        Value::String("split".to_string()),
        Value::String("native_dataset_split".to_string()),
    ];
    let method_split_fold = vec![
        Value::String("__method__".to_string()),
        Value::String("split".to_string()),
        Value::String("native_dataset_split_fold".to_string()),
    ];
    let method_push = vec![
        Value::String("__method__".to_string()),
        Value::String("push_data".to_string()),
        Value::String("native_dataset_push_data".to_string()),
    ];
    let method_concat = vec![
        Value::String("__method__".to_string()),
        Value::String("concat".to_string()),
        Value::String("native_dataset_concat".to_string()),
    ];
    let row_split_fold = vec![
        Value::String("native_dataset_split_fold".to_string()),
        Value::String("split".to_string()),
    ];
    Value::Array(Rc::new(RefCell::new(vec![
        Value::Array(Rc::new(RefCell::new(row_split))),
        Value::Array(Rc::new(RefCell::new(row_split_fold))),
        Value::Array(Rc::new(RefCell::new(row_push))),
        Value::Array(Rc::new(RefCell::new(row_concat))),
        Value::Array(Rc::new(RefCell::new(method_split))),
        Value::Array(Rc::new(RefCell::new(method_split_fold))),
        Value::Array(Rc::new(RefCell::new(method_push))),
        Value::Array(Rc::new(RefCell::new(method_concat))),
    ])))
}

/// Binary ops on plugin opaque values from the VM (`a + b`, `@`, …): `(left, right, op)`.
/// `op` is `"add"` | `"sub"` | `"mul"` | `"matmul"`. Type checks stay in the existing natives.
pub fn native_opaque_binop(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }
    let Value::String(ref op) = args[2] else {
        return Value::Null;
    };
    let pair = &[args[0].clone(), args[1].clone()];
    match op.as_str() {
        "add" => native_add(pair),
        "sub" => native_sub(pair),
        "mul" => native_mul(pair),
        "matmul" => native_matmul(pair),
        _ => Value::Null,
    }
}

/// Transpose tensor
/// transpose(tensor)
pub fn native_transpose(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    match t.transpose() {
        Ok(result) => crate::runtime::tensor_to_value(result),
        Err(_) => Value::Null,
    }
}

/// Сумма по последней оси с keepdim (2D: сумма по строкам → форма [rows, 1]).
/// Встроенные `sum`/`average` в VM вызывают это через `native_plugin_call(_, "sum"|"mean")`.
pub fn native_sum(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    if t.shape().is_empty() {
        return Value::Null;
    }
    let out = crate::ops::sum_last_axis_keepdim(&t);
    crate::runtime::tensor_to_value(out)
}

/// Среднее по последней оси с keepdim (2D: по строкам → [rows, 1]).
pub fn native_mean(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    if t.shape().is_empty() {
        return Value::Null;
    }
    let out = crate::ops::mean_last_axis_keepdim(&t);
    crate::runtime::tensor_to_value(out)
}

/// Find index(es) of maximum element(s)
/// max_idx(tensor) -> array of indices
/// For 1D tensors: returns array with single element
/// For multi-dimensional tensors: returns array with indices for each slice along first dimension
pub fn native_max_idx(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    match t.max_idx() {
        Ok(indices) => {
            let indices_values: Vec<Value> = indices.iter().map(|&idx| Value::Number(idx as f64)).collect();
            Value::Array(Rc::new(RefCell::new(indices_values)))
        }
        Err(_) => Value::Null,
    }
}

/// Find index(es) of minimum element(s)
/// min_idx(tensor) -> array of indices
/// For 1D tensors: returns array with single element
/// For multi-dimensional tensors: returns array with indices for each slice along first dimension
pub fn native_min_idx(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    match t.min_idx() {
        Ok(indices) => {
            let indices_values: Vec<Value> = indices.iter().map(|&idx| Value::Number(idx as f64)).collect();
            Value::Array(Rc::new(RefCell::new(indices_values)))
        }
        Err(_) => Value::Null,
    }
}

/// Create a new computational graph
/// graph()
pub fn native_graph(args: &[Value]) -> Value {
    // Module-style call ml.graph() passes receiver as first arg; native expects 0 args
    let args = if args.len() == 1
        && matches!(&args[0], Value::Object(_) | Value::ObjectPtr(_))
    {
        &args[1..]
    } else {
        args
    };
    if !args.is_empty() {
        return Value::Null;
    }

    let graph = Graph::new();
    crate::runtime::graph_to_value(graph)
}

/// Add an input placeholder node to the graph
/// graph_add_input(graph) -> node_id
pub fn native_graph_add_input(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let node_id = graph.borrow_mut().add_input();
    Value::Number(node_id as f64)
}

/// Add an operation node to the graph
/// graph_add_op(graph, op_name, input_node_ids) -> node_id
pub fn native_graph_add_op(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let op_name = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Value::Null,
    };

    let input_ids_array = match &args[2] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut input_ids = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => {
                        let id = *n as i64;
                        if id < 0 {
                            return Value::Null;
                        }
                        input_ids.push(id as usize);
                    }
                    _ => return Value::Null,
                }
            }
            input_ids
        }
        _ => return Value::Null,
    };

    // Parse operation type
    let op = match op_name {
        "add" => OpType::Add,
        "sub" => OpType::Sub,
        "mul" => OpType::Mul,
        "matmul" => OpType::MatMul,
        "transpose" => OpType::Transpose,
        "sum" => OpType::Sum,
        "mean" => OpType::Mean,
        _ => return Value::Null,
    };

    let node_res = {
        let mut g = graph.borrow_mut();
        g.add_op(op, input_ids_array)
    };
    match node_res {
        Ok(node_id) => Value::Number(node_id as f64),
        Err(_) => Value::Null,
    }
}

/// Execute forward pass through the graph
/// graph_forward(graph, input_tensors)
pub fn native_graph_forward(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let input_tensors_array = match &args[1] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut tensors = Vec::new();
            for val in arr_ref.iter() {
                let Some(t) = crate::runtime::tensor_data_clone(val) else {
                    return Value::Null;
                };
                tensors.push(t);
            }
            tensors
        }
        _ => return Value::Null,
    };

    let fwd = {
        let mut g = graph.borrow_mut();
        g.forward(input_tensors_array)
    };
    match fwd {
        Ok(_) => Value::Null, // Forward returns void
        Err(_) => Value::Null, // Return Null on error (could be improved with error handling)
    }
}

/// Get the output tensor of a node (after forward pass)
/// graph_get_output(graph, node_id) -> tensor
pub fn native_graph_get_output(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let node_id = match &args[1] {
        Value::Number(n) => {
            let id = *n as i64;
            if id < 0 {
                return Value::Null;
            }
            id as usize
        }
        _ => return Value::Null,
    };

    let out = {
        let g = graph.borrow();
        g.get_output(node_id)
    };
    match out {
        Ok(tensor) => crate::runtime::tensor_to_value(tensor),
        Err(_) => Value::Null,
    }
}

/// Execute backward pass to compute gradients
/// graph_backward(graph, output_node_id)
pub fn native_graph_backward(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let output_node_id = match &args[1] {
        Value::Number(n) => {
            let id = *n as i64;
            if id < 0 {
                return Value::Null;
            }
            id as usize
        }
        _ => return Value::Null,
    };

    let bw = {
        let mut g = graph.borrow_mut();
        g.backward(output_node_id)
    };
    match bw {
        Ok(_) => Value::Null, // Backward returns void
        Err(_) => Value::Null, // Return Null on error
    }
}

/// Get the gradient of a node (after backward pass)
/// graph_get_gradient(graph, node_id) -> tensor
pub fn native_graph_get_gradient(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let node_id = match &args[1] {
        Value::Number(n) => {
            let id = *n as i64;
            if id < 0 {
                return Value::Null;
            }
            id as usize
        }
        _ => return Value::Null,
    };

    let grad = {
        let g = graph.borrow();
        g.get_gradient(node_id)
    };
    match grad {
        Ok(tensor) => crate::runtime::tensor_to_value(tensor),
        Err(_) => Value::Null,
    }
}

/// Zero all gradients in the graph
/// graph_zero_grad(graph)
pub fn native_graph_zero_grad(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    graph.borrow_mut().zero_grad();
    Value::Null
}

/// Set whether a node requires gradients
/// graph_set_requires_grad(graph, node_id, requires_grad)
pub fn native_graph_set_requires_grad(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let node_id = match &args[1] {
        Value::Number(n) => {
            let id = *n as i64;
            if id < 0 {
                return Value::Null;
            }
            id as usize
        }
        _ => return Value::Null,
    };

    let requires_grad = match &args[2] {
        Value::Bool(b) => *b,
        _ => return Value::Null,
    };

    let sr = {
        let mut g = graph.borrow_mut();
        g.set_requires_grad(node_id, requires_grad)
    };
    match sr {
        Ok(_) => Value::Null,
        Err(_) => Value::Null,
    }
}

/// Create a new Linear Regression model
/// linear_regression(feature_count) -> model
pub fn native_linear_regression(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let feature_count = match &args[0] {
        Value::Number(n) => {
            let count = *n as i64;
            if count <= 0 {
                return Value::Null;
            }
            count as usize
        }
        _ => return Value::Null,
    };

    match LinearRegression::new(feature_count) {
        Ok(model) => crate::runtime::linear_regression_to_value(model),
        Err(_) => Value::Null,
    }
}

/// Predict outputs for given features
/// lr_predict(model, features) -> predictions
pub fn native_lr_predict(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let model = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::LinearRegression as u8 => crate::runtime::get_linear_regression(*id).expect("lr"),
        _ => return Value::Null,
    };

    let Some(features) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    let pred = {
        let m = model.borrow();
        m.predict(&features)
    };
    match pred {
        Ok(predictions) => crate::runtime::tensor_to_value(predictions),
        Err(_) => Value::Null,
    }
}

/// Train the model
/// lr_train(model, x, y, epochs, lr) -> loss_history
pub fn native_lr_train(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    if args.len() != 5 {
        set_native_error(format!("ml.lr_train expects 5 arguments (model, x, y, epochs, lr), got {}", args.len()));
        return Value::Null;
    }

    let model = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::LinearRegression as u8 => crate::runtime::get_linear_regression(*id).expect("lr"),
        _ => {
            set_native_error("ml.lr_train: first argument must be a LinearRegression model (from ml.linear_regression)".to_string());
            return Value::Null;
        }
    };

    let Some(x) = crate::runtime::tensor_data_clone(&args[1]) else {
        set_native_error("ml.lr_train: second argument (x) must be a Tensor".to_string().to_string());
        return Value::Null;
    };

    let Some(y) = crate::runtime::tensor_data_clone(&args[2]) else {
        set_native_error("ml.lr_train: third argument (y) must be a Tensor".to_string().to_string());
        return Value::Null;
    };

    let epochs = match &args[3] {
        Value::Number(n) => {
            let e = *n as i64;
            if e <= 0 {
                set_native_error("ml.lr_train: epochs must be a positive number".to_string());
                return Value::Null;
            }
            e as usize
        }
        _ => {
            set_native_error("ml.lr_train: fourth argument (epochs) must be a number".to_string());
            return Value::Null;
        }
    };

    let lr = match &args[4] {
        Value::Number(n) => {
            let rate = *n as f64;
            if rate <= 0.0 {
                set_native_error("ml.lr_train: learning rate must be positive".to_string());
                return Value::Null;
            }
            rate as f32
        }
        _ => {
            set_native_error("ml.lr_train: fifth argument (lr) must be a number".to_string());
            return Value::Null;
        }
    };

    let train_res = {
        let mut m = model.borrow_mut();
        m.train(&x, &y, epochs, lr)
    };
    match train_res {
        Ok(loss_history) => {
            let loss_values: Vec<Value> = loss_history.iter().map(|&v| Value::Number(v as f64)).collect();
            Value::Array(Rc::new(RefCell::new(loss_values)))
        }
        Err(e) => {
            set_native_error(format!("ml.lr_train failed: {}", e));
            Value::Null
        }
    }
}

/// Evaluate the model (compute MSE)
/// lr_evaluate(model, x, y) -> mse
pub fn native_lr_evaluate(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }

    let model = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::LinearRegression as u8 => crate::runtime::get_linear_regression(*id).expect("lr"),
        _ => return Value::Null,
    };

    let Some(x) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    let Some(y) = crate::runtime::tensor_data_clone(&args[2]) else {
        return Value::Null;
    };

    let ev = {
        let m = model.borrow();
        m.evaluate(&x, &y)
    };
    match ev {
        Ok(mse) => Value::Number(mse as f64),
        Err(_) => Value::Null,
    }
}

/// Create a new SGD optimizer
/// sgd(learning_rate) -> optimizer
pub fn native_sgd(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let learning_rate = match &args[0] {
        Value::Number(n) => {
            let lr = *n as f64;
            if lr <= 0.0 {
                return Value::Null;
            }
            lr as f32
        }
        _ => return Value::Null,
    };

    match SGD::new(learning_rate) {
        Ok(optimizer) => crate::runtime::sgd_to_value(optimizer),
        Err(_) => Value::Null,
    }
}

/// Perform one optimization step
/// sgd_step(optimizer, graph, param_node_ids) -> null
pub fn native_sgd_step(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }

    let optimizer = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Sgd as u8 => crate::runtime::get_sgd(*id).expect("sgd"),
        _ => return Value::Null,
    };

    let graph = match &args[1] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let param_ids = match &args[2] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut param_ids = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => {
                        let id = *n as i64;
                        if id < 0 {
                            return Value::Null;
                        }
                        param_ids.push(id as usize);
                    }
                    _ => return Value::Null,
                }
            }
            param_ids
        }
        _ => return Value::Null,
    };

    // Get learning rate from optimizer
    let lr = optimizer.borrow().lr;
    
    // Update parameters in Graph
    let mut graph_ref = graph.borrow_mut();
    
    for &node_id in &param_ids {
        // Get gradient for this parameter
        let grad = match graph_ref.get_gradient(node_id) {
            Ok(g) => g,
            Err(_) => continue, // Skip if no gradient
        };
        
        // Get current parameter value
        let param = match graph_ref.get_output(node_id) {
            Ok(p) => p,
            Err(_) => continue, // Skip if no value
        };
        
        // SGD update: param_new = param_old - lr * grad
        use crate::ops;
        let update = ops::scalar_mul(&grad, -lr);
        let new_param = ops::add(&param, &update);
        
        // Update parameter value in Graph
        graph_ref.nodes[node_id].value = Some(new_param);
    }
    
    Value::Null
}

/// Zero gradients in the graph (convenience function)
/// sgd_zero_grad(graph) -> null
pub fn native_sgd_zero_grad(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let graph = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    graph.borrow_mut().zero_grad();
    Value::Null
}

/// Create a new Adam optimizer
/// adam(learning_rate) -> optimizer
/// adam(learning_rate, beta1, beta2, epsilon) -> optimizer
pub fn native_adam(args: &[Value]) -> Value {
    if args.len() < 1 || args.len() > 4 {
        return Value::Null;
    }

    let learning_rate = match &args[0] {
        Value::Number(n) => {
            let lr = *n as f64;
            if lr <= 0.0 {
                return Value::Null;
            }
            lr as f32
        }
        _ => return Value::Null,
    };

    let beta1 = if args.len() >= 2 {
        match &args[1] {
            Value::Number(n) => {
                let b = *n as f64;
                if b < 0.0 || b >= 1.0 {
                    return Value::Null;
                }
                b as f32
            }
            _ => return Value::Null,
        }
    } else {
        0.9
    };

    let beta2 = if args.len() >= 3 {
        match &args[2] {
            Value::Number(n) => {
                let b = *n as f64;
                if b < 0.0 || b >= 1.0 {
                    return Value::Null;
                }
                b as f32
            }
            _ => return Value::Null,
        }
    } else {
        0.999
    };

    let epsilon = if args.len() >= 4 {
        match &args[3] {
            Value::Number(n) => {
                let e = *n as f64;
                if e <= 0.0 {
                    return Value::Null;
                }
                e as f32
            }
            _ => return Value::Null,
        }
    } else {
        1e-8
    };

    match Adam::with_params(learning_rate, beta1, beta2, epsilon) {
        Ok(optimizer) => crate::runtime::adam_to_value(optimizer),
        Err(_) => Value::Null,
    }
}

/// Perform one optimization step with Adam
/// adam_step(optimizer, graph, param_node_ids) -> null
pub fn native_adam_step(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }

    let optimizer = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Adam as u8 => crate::runtime::get_adam(*id).expect("adam"),
        _ => return Value::Null,
    };

    let graph = match &args[1] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Graph as u8 => crate::runtime::get_graph(*id).expect("graph"),
        _ => return Value::Null,
    };

    let param_ids = match &args[2] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut param_ids = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::Number(n) => {
                        let id = *n as i64;
                        if id < 0 {
                            return Value::Null;
                        }
                        param_ids.push(id as usize);
                    }
                    _ => return Value::Null,
                }
            }
            param_ids
        }
        _ => return Value::Null,
    };

    // Get optimizer parameters
    let mut opt_ref = optimizer.borrow_mut();
    let lr = opt_ref.lr;
    let beta1 = opt_ref.beta1;
    let beta2 = opt_ref.beta2;
    let epsilon = opt_ref.epsilon;
    
    // Increment step count
    opt_ref.step_count += 1;
    let t = opt_ref.step_count as f32;
    
    // Bias correction coefficients: 1 - beta^t
    let bias_correction1 = 1.0 - beta1.powf(t);
    let bias_correction2 = 1.0 - beta2.powf(t);
    
    // Update parameters in Graph
    let mut graph_ref = graph.borrow_mut();
    use crate::ops;
    const MAX_GRAD_NORM: f32 = 1.0; // Gradient clipping threshold
    
    for &node_id in &param_ids {
        // Get gradient for this parameter
        let grad = match graph_ref.get_gradient(node_id) {
            Ok(g) => g,
            Err(_) => continue, // Skip if no gradient
        };
        
        // Get current parameter value
        let param = match graph_ref.get_output(node_id) {
            Ok(p) => p,
            Err(_) => continue, // Skip if no value
        };
        
        // Check gradient for NaN and Inf
        let grad_arr = grad.data();
        let mut has_nan = false;
        let mut grad_norm_sq = 0.0;
        for val in grad_arr.iter() {
            if val.is_nan() || val.is_infinite() {
                has_nan = true;
                break;
            }
            grad_norm_sq += val * val;
        }
        
        // Skip update if gradient contains NaN or Inf
        if has_nan {
            continue;
        }
        
        // Gradient clipping
        let grad_norm = grad_norm_sq.sqrt();
        let clipped_grad = if grad_norm > MAX_GRAD_NORM {
            ops::scalar_mul(&grad, MAX_GRAD_NORM / grad_norm)
        } else {
            grad.clone()
        };
        
        // Initialize moments if needed and get references
        let grad_shape = grad.shape().to_vec();
        let m_old = opt_ref.graph_m.entry(node_id).or_insert_with(|| {
            Tensor::zeros(grad_shape.clone())
        }).clone();
        let v_old = opt_ref.graph_v.entry(node_id).or_insert_with(|| {
            Tensor::zeros(grad_shape)
        }).clone();
        
        // Update moments
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        let m_scaled = ops::scalar_mul(&m_old, beta1);
        let m_grad = ops::scalar_mul(&clipped_grad, 1.0 - beta1);
        let m_new = ops::add(&m_scaled, &m_grad);
        
        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        let grad_squared = ops::mul(&clipped_grad, &clipped_grad);
        let v_scaled = ops::scalar_mul(&v_old, beta2);
        let v_grad = ops::scalar_mul(&grad_squared, 1.0 - beta2);
        let v_new = ops::add(&v_scaled, &v_grad);
        
        // Bias correction
        // m_hat = m_t / (1 - beta1^t)
        let m_hat = ops::scalar_div(&m_new, bias_correction1);
        // v_hat = v_t / (1 - beta2^t)
        let v_hat = ops::scalar_div(&v_new, bias_correction2);
        
        // Compute update: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        let v_hat_sqrt = ops::sqrt(&v_hat);
        // Create epsilon tensor of the same shape
        let epsilon_tensor = Tensor::from_slice(
            &vec![epsilon; v_hat_sqrt.numel()],
            v_hat_sqrt.shape()
        );
        let v_hat_sqrt_eps = ops::add(&v_hat_sqrt, &epsilon_tensor);
        let update_ratio = ops::div(&m_hat, &v_hat_sqrt_eps);
        let update = ops::scalar_mul(&update_ratio, -lr);
        
        // Compute new parameter value
        let new_param = ops::add(&param, &update);
        
        // Check result for NaN before updating
        let new_param_arr = new_param.data();
        let mut has_nan_result = false;
        for val in new_param_arr.iter() {
            if val.is_nan() || val.is_infinite() {
                has_nan_result = true;
                break;
            }
        }
        
        // Update only if result is valid
        if !has_nan_result {
            // Update parameter value in Graph
            graph_ref.nodes[node_id].value = Some(new_param);
            // Update moments in optimizer
            opt_ref.graph_m.insert(node_id, m_new);
            opt_ref.graph_v.insert(node_id, v_new);
        }
    }
    
    Value::Null
}

/// Compute MSE loss
/// mse_loss(y_pred, y_true) -> loss_tensor
pub fn native_mse_loss(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match mse_loss(&y_pred, &y_true) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Compute Cross Entropy loss
/// cross_entropy_loss(logits, class_indices) -> loss_tensor
/// DEPRECATED: This function is deprecated. Use sparse_softmax_cross_entropy_loss directly.
/// For one-hot targets, use categorical_cross_entropy_loss.
pub fn native_cross_entropy_loss(_args: &[Value]) -> Value {
    // This function is deprecated - redirect to sparse implementation
    // Users should use sparse_softmax_cross_entropy_loss or categorical_cross_entropy_loss directly
    Value::Null
}

/// Compute Binary Cross Entropy loss
/// binary_cross_entropy_loss(y_pred, y_true) -> loss_tensor
pub fn native_binary_cross_entropy_loss(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match binary_cross_entropy_loss(&y_pred, &y_true) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Compute Mean Absolute Error loss
/// mae_loss(y_pred, y_true) -> loss_tensor
pub fn native_mae_loss(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match mae_loss(&y_pred, &y_true) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Compute Huber loss
/// huber_loss(y_pred, y_true, delta) -> loss_tensor
/// If delta is not provided, defaults to 1.0
pub fn native_huber_loss(args: &[Value]) -> Value {
    if args.len() < 2 || args.len() > 3 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    // Delta parameter (default to 1.0 if not provided)
    let delta = if args.len() == 3 {
        match &args[2] {
            Value::Number(n) => {
                let d = *n as f32;
                if d <= 0.0 {
                    return Value::Null;
                }
                d
            }
            _ => return Value::Null,
        }
    } else {
        1.0
    };

    match huber_loss(&y_pred, &y_true, delta) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Compute Hinge loss
/// hinge_loss(y_pred, y_true) -> loss_tensor
pub fn native_hinge_loss(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match hinge_loss(&y_pred, &y_true) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Compute KL Divergence
/// kl_divergence(y_pred, y_true) -> loss_tensor
pub fn native_kl_divergence(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match kl_divergence(&y_pred, &y_true) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Compute Smooth L1 loss
/// smooth_l1_loss(y_pred, y_true) -> loss_tensor
pub fn native_smooth_l1_loss(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(y_pred) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(y_true) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match smooth_l1_loss(&y_pred, &y_true) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

/// Create a dataset from a table (`ml.dataset.from_table` / `dataset.from_table`).
/// `table` is `AbiValue::Table` bridged to `Value::Array([headers, rows])`.
pub fn native_dataset(args: &[Value]) -> Value {
    if args.len() != 3 {
        use crate::native_error::set_native_error;
        set_native_error(format!(
            "ml.dataset.from_table expects 3 arguments (table, feature_cols, target_cols), got {}",
            args.len()
        ));
        return Value::Null;
    }

    let (headers, rows) = match Dataset::parse_abi_table_from_value(&args[0]) {
        Ok(x) => x,
        Err(e) => {
            use crate::native_error::set_native_error;
            set_native_error(e);
            return Value::Null;
        }
    };

    let feature_columns_array = match &args[1] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut columns = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::String(s) => columns.push(s.clone()),
                    _ => return Value::Null,
                }
            }
            columns
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("ml.dataset.from_table: second argument (feature_cols) must be an array of strings".to_string());
            return Value::Null;
        }
    };

    let target_columns_array = match &args[2] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut columns = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::String(s) => columns.push(s.clone()),
                    _ => {
                        use crate::native_error::set_native_error;
                        set_native_error("ml.dataset.from_table: target_cols must be an array of strings".to_string());
                        return Value::Null;
                    }
                }
            }
            columns
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("ml.dataset.from_table: third argument (target_cols) must be an array of strings".to_string());
            return Value::Null;
        }
    };

    match Dataset::from_abi_table(&headers, &rows, &feature_columns_array, &target_columns_array) {
        Ok(dataset) => crate::runtime::dataset_to_value(dataset),
        Err(e) => {
            use crate::native_error::set_native_error;
            set_native_error(format!("ml.dataset.from_table failed: {}", e));
            Value::Null
        }
    }
}

/// Get features tensor from dataset
/// dataset_features(dataset) -> features_tensor
pub fn native_dataset_features(args: &[Value]) -> Value {
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error(format!("ml.dataset_features expects 1 argument (dataset), got {}", args.len()));
        return Value::Null;
    }

    let dataset = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => crate::runtime::get_dataset(*id).expect("dataset"),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("ml.dataset_features: argument must be a Dataset (from ml.dataset)".to_string());
            return Value::Null;
        }
    };

    let features = dataset.borrow().features().clone();
    crate::runtime::tensor_to_value(features)
}

/// Get targets tensor from dataset
/// dataset_targets(dataset) -> targets_tensor
pub fn native_dataset_targets(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let dataset = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => crate::runtime::get_dataset(*id).expect("dataset"),
        _ => return Value::Null,
    };

    let targets = dataset.borrow().targets().clone();
    crate::runtime::tensor_to_value(targets)
}

/// `dataset.from_tensors` / `dataset_from_tensors`: (features_tensor, targets_tensor) -> dataset.
pub fn native_dataset_from_tensors(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    if args.len() != 2 {
        set_native_error(format!(
            "ml.dataset.from_tensors expects 2 arguments (features, targets), got {}",
            args.len()
        ));
        return Value::Null;
    }

    let Some(ft) = crate::runtime::tensor_from_value_flexible(&args[0]) else {
        let extra = crate::runtime::features_array_mixed_rows_hint(&args[0])
            .unwrap_or("Expected a Tensor handle or a rectangular nested array of numbers (e.g. [N][3072] for N rows of pixels).");
        set_native_error(format!(
            "ml.dataset.from_tensors: features must be a Tensor or nested array of numbers. {}",
            extra
        ));
        return Value::Null;
    };
    let Some(tt) = crate::runtime::tensor_from_value_flexible(&args[1]) else {
        set_native_error(
            "ml.dataset.from_tensors: targets must be a Tensor or nested array of numbers".to_string(),
        );
        return Value::Null;
    };

    match Dataset::from_tensors(ft, tt) {
        Ok(d) => crate::runtime::dataset_to_value(d),
        Err(e) => {
            set_native_error(e);
            Value::Null
        }
    }
}

/// `dataset.concat(other)` — append rows from `other` into this dataset in place.
/// Args: `[receiver, other_dataset]`.
pub fn native_dataset_concat(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    if args.len() != 2 {
        set_native_error(format!(
            "dataset.concat expects 2 arguments (dataset, other), got {}",
            args.len()
        ));
        return Value::Null;
    }
    let receiver = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => {
            crate::runtime::get_dataset(*id).expect("dataset")
        }
        _ => {
            set_native_error("dataset.concat: receiver must be a Dataset".to_string());
            return Value::Null;
        }
    };
    let other_ref = match &args[1] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => {
            crate::runtime::get_dataset(*id).expect("dataset")
        }
        _ => {
            set_native_error("dataset.concat: second argument must be a Dataset".to_string());
            return Value::Null;
        }
    };
    let other_ds: Dataset = other_ref.borrow().clone();
    let res = {
        let mut r = receiver.borrow_mut();
        r.concat_in_place(&other_ds)
    };
    match res {
        Ok(()) => Value::Null,
        Err(e) => {
            set_native_error(e);
            Value::Null
        }
    }
}

/// `dataset.push_data(features=..., targets=...)` — append rows to this dataset in place.
/// Args: `[receiver, features, targets]`.
pub fn native_dataset_push_data(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    if args.len() != 3 {
        set_native_error(format!(
            "dataset.push_data expects 3 arguments (dataset, features, targets), got {}",
            args.len()
        ));
        return Value::Null;
    }
    let receiver = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => {
            crate::runtime::get_dataset(*id).expect("dataset")
        }
        _ => {
            set_native_error("dataset.push_data: receiver must be a Dataset".to_string());
            return Value::Null;
        }
    };
    let Some(ft) = crate::runtime::tensor_from_value_flexible(&args[1]) else {
        set_native_error(
            "dataset.push_data: features must be a Tensor or nested array of numbers".to_string(),
        );
        return Value::Null;
    };
    let Some(tt) = crate::runtime::tensor_from_value_flexible(&args[2]) else {
        set_native_error(
            "dataset.push_data: targets must be a Tensor or nested array of numbers".to_string(),
        );
        return Value::Null;
    };
    let res = {
        let mut r = receiver.borrow_mut();
        r.push_tensor_rows(ft, tt)
    };
    match res {
        Ok(()) => Value::Null,
        Err(e) => {
            set_native_error(e);
            Value::Null
        }
    }
}

/// dataset_len(dataset) -> number of samples (same as features.shape[0])
pub fn native_dataset_len(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    if args.len() != 1 {
        set_native_error(format!("ml.dataset_len expects 1 argument (dataset), got {}", args.len()));
        return Value::Null;
    }
    let dataset = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => {
            crate::runtime::get_dataset(*id).expect("dataset")
        }
        _ => {
            set_native_error("ml.dataset_len: argument must be a Dataset".to_string());
            return Value::Null;
        }
    };
    let n = dataset.borrow().batch_size();
    Value::Number(n as f64)
}

/// ABI: whether `value` is a Dataset opaque that the VM can iterate (`for ... in`) using
/// `dataset_len` + indexed access via `native_plugin_call` / `dataset_get`.
pub fn native_dataset_supports_iteration(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Bool(false);
    }
    match &args[0] {
        Value::PluginOpaque { tag, .. } if *tag == MlValueKind::Dataset as u8 => Value::Bool(true),
        Value::PluginOpaque { tag, .. } if *tag == MlValueKind::DatasetCatalog as u8 => Value::Bool(false),
        _ => Value::Bool(false),
    }
}

/// dataset_get(dataset, index) -> [features_row, targets_row] (each rank-1 tensors)
pub fn native_dataset_get(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    use std::cell::RefCell;
    use std::rc::Rc;

    if args.len() != 2 {
        set_native_error(format!(
            "ml.dataset_get expects 2 arguments (dataset, index), got {}",
            args.len()
        ));
        return Value::Null;
    }

    let dataset = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => {
            crate::runtime::get_dataset(*id).expect("dataset")
        }
        _ => {
            set_native_error("ml.dataset_get: first argument must be a Dataset".to_string());
            return Value::Null;
        }
    };

    let idx = match &args[1] {
        Value::Number(n) => {
            let i = *n as i64;
            if i < 0 {
                set_native_error("ml.dataset_get: index must be non-negative".to_string());
                return Value::Null;
            }
            i as usize
        }
        _ => {
            set_native_error("ml.dataset_get: index must be a number".to_string());
            return Value::Null;
        }
    };

    let ds = dataset.borrow();
    let f_row = match ds.features().get_row(idx) {
        Ok(t) => t,
        Err(e) => {
            set_native_error(e);
            return Value::Null;
        }
    };
    let t_row = match ds.targets().get_row(idx) {
        Ok(t) => t,
        Err(e) => {
            set_native_error(e);
            return Value::Null;
        }
    };

    Value::Array(Rc::new(RefCell::new(vec![
        crate::runtime::tensor_to_value(f_row),
        crate::runtime::tensor_to_value(t_row),
    ])))
}

/// Шесть опций sklearn-style: `test_size`, `train_size`, `shuffle`, `random_state`, `stratify`, `return_indices`.
fn native_dataset_split_on_materialized(dataset: &Dataset, args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    use std::cell::RefCell;
    use std::rc::Rc;

    if args.len() != 6 {
        set_native_error(format!(
            "dataset.split expects 6 option arguments, got {}",
            args.len()
        ));
        return Value::Null;
    }

    let test_size = match &args[0] {
        Value::Null => None,
        Value::Number(n) => Some(*n),
        _ => {
            set_native_error("dataset.split: test_size must be null or number".to_string());
            return Value::Null;
        }
    };
    let train_size = match &args[1] {
        Value::Null => None,
        Value::Number(n) => Some(*n),
        _ => {
            set_native_error("dataset.split: train_size must be null or number".to_string());
            return Value::Null;
        }
    };
    let shuffle = match &args[2] {
        Value::Null => crate::dataset_split_abi::DEFAULT_SPLIT_SHUFFLE,
        Value::Bool(b) => *b,
        _ => {
            set_native_error("dataset.split: shuffle must be null or bool".to_string());
            return Value::Null;
        }
    };
    let random_state = match &args[3] {
        Value::Null => None,
        Value::Number(n) => {
            if *n < 0.0 || n.fract() != 0.0 {
                set_native_error(
                    "dataset.split: random_state must be null or non-negative integer".to_string(),
                );
                return Value::Null;
            }
            Some(*n as u64)
        }
        _ => {
            set_native_error("dataset.split: random_state must be null or number".to_string());
            return Value::Null;
        }
    };
    let stratify = match &args[4] {
        Value::Null => crate::dataset_split_abi::DEFAULT_SPLIT_STRATIFY,
        Value::Bool(b) => *b,
        _ => {
            set_native_error("dataset.split: stratify must be null or bool".to_string());
            return Value::Null;
        }
    };
    let return_indices = match &args[5] {
        Value::Null => crate::dataset_split_abi::DEFAULT_SPLIT_RETURN_INDICES,
        Value::Bool(b) => *b,
        _ => {
            set_native_error("dataset.split: return_indices must be null or bool".to_string());
            return Value::Null;
        }
    };

    let result = match dataset.split(
        test_size,
        train_size,
        shuffle,
        random_state,
        stratify,
        return_indices,
    ) {
        Ok(r) => r,
        Err(e) => {
            set_native_error(e);
            return Value::Null;
        }
    };

    let train_v = crate::runtime::dataset_to_value(result.train);
    let test_v = crate::runtime::dataset_to_value(result.test);
    if return_indices {
        let tr = result.train_indices.unwrap_or_default();
        let te = result.test_indices.unwrap_or_default();
        let tr_a: Vec<Value> = tr.iter().map(|&i| Value::Number(i as f64)).collect();
        let te_a: Vec<Value> = te.iter().map(|&i| Value::Number(i as f64)).collect();
        Value::Array(Rc::new(RefCell::new(vec![
            train_v,
            test_v,
            Value::Array(Rc::new(RefCell::new(tr_a))),
            Value::Array(Rc::new(RefCell::new(te_a))),
        ])))
    } else {
        Value::Array(Rc::new(RefCell::new(vec![train_v, test_v])))
    }
}

/// `dataset.split(test_size, train_size, shuffle, random_state, stratify, return_indices)` — VM passes
/// receiver first, then six options (null/bool/number). Kwargs / compiler metadata: ML repo crate
/// `crates/datacode_ml_compiler/ml_native_named_args.json` key `native_dataset_split` and
/// [`crate::DATASET_SPLIT_NAMED_ARG_NAMES`]. Omitted bool kwargs arrive as `Null`; defaults are in
/// [`crate::dataset_split_abi`].
pub fn native_dataset_split(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;

    if args.len() != 7 {
        set_native_error(format!(
            "dataset.split expects 7 arguments (dataset + 6 options), got {}",
            args.len()
        ));
        return Value::Null;
    }

    let dataset = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 => {
            crate::runtime::get_dataset(*id).expect("dataset")
        }
        _ => {
            set_native_error("dataset.split: first argument must be a Dataset".to_string());
            return Value::Null;
        }
    };

    let borrowed = dataset.borrow();
    native_dataset_split_on_materialized(&*borrowed, &args[1..7])
}

/// Convert integer labels to one-hot encoding (batch)
/// onehots(labels, num_classes?) -> tensor [N, num_classes]
/// If num_classes is not provided, it is determined as max(label) + 1
pub fn native_onehots(args: &[Value]) -> Value {
    if args.is_empty() || args.len() > 2 {
        return Value::Null;
    }

    // Extract labels tensor
    let Some(labels_tensor) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    // Ensure tensor is on CPU for processing
    let labels_cpu = match labels_tensor.to_cpu() {
        Ok(t) => t,
        Err(_) => return Value::Null,
    };

    // Extract labels from tensor - support both [N] and [N, 1] shapes
    let labels: Vec<usize> = if labels_cpu.ndim() == 1 {
        // Shape [N]
        labels_cpu.as_slice().iter().map(|&x| x as usize).collect()
    } else if labels_cpu.ndim() == 2 && labels_cpu.shape[1] == 1 {
        // Shape [N, 1]
        labels_cpu.as_slice().iter().map(|&x| x as usize).collect()
    } else {
        return Value::Null;
    };

    if labels.is_empty() {
        return Value::Null;
    }

    // Determine number of classes
    let num_classes = if args.len() == 2 {
        // Provided as argument
        match &args[1] {
            Value::Number(n) => {
                let count = *n as i64;
                if count <= 0 {
                    return Value::Null;
                }
                count as usize
            }
            _ => return Value::Null,
        }
    } else {
        // Auto-detect from max label + 1
        let max_label = labels.iter().max().copied().unwrap_or(0);
        max_label + 1
    };

    // Validate that all labels are within range [0, num_classes)
    for &label in &labels {
        if label >= num_classes {
            return Value::Null;
        }
    }

    // Create one-hot encoded tensor
    let n = labels.len();
    let mut onehot_data = vec![0.0; n * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        onehot_data[i * num_classes + label] = 1.0;
    }

    match Tensor::new(onehot_data, vec![n, num_classes]) {
        Ok(tensor) => crate::runtime::tensor_to_value(tensor),
        Err(_) => Value::Null,
    }
}

/// Single row one-hot: `one_hot(class_index, num_classes)` -> tensor [1, num_classes]
pub fn native_one_hot(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;

    if args.len() != 2 {
        set_native_error(format!(
            "ml.one_hot expects 2 arguments (class_index, num_classes), got {}",
            args.len()
        ));
        return Value::Null;
    }
    let (Value::Number(idx_n), Value::Number(nc_n)) = (&args[0], &args[1]) else {
        set_native_error(
            "ml.one_hot: class_index and num_classes must be numbers (e.g. pass a scalar label, not an array or tensor row)".to_string(),
        );
        return Value::Null;
    };
    let num_classes = *nc_n as i64;
    let index = *idx_n as i64;
    if num_classes <= 0 {
        set_native_error("ml.one_hot: num_classes must be positive".to_string());
        return Value::Null;
    }
    if index < 0 || index >= num_classes {
        set_native_error(format!(
            "ml.one_hot: class_index {} out of range for num_classes {}",
            index, num_classes
        ));
        return Value::Null;
    }
    let nc = num_classes as usize;
    let ix = index as usize;
    let mut data = vec![0.0f32; nc];
    data[ix] = 1.0;
    match Tensor::new(data, vec![1, nc]) {
        Ok(tensor) => crate::runtime::tensor_to_value(tensor),
        Err(_) => Value::Null,
    }
}

/// Create a Linear layer
/// linear_layer(in_features, out_features) or linear_layer(in, out, use_bias: 0|1)
pub fn native_linear_layer(args: &[Value]) -> Value {
    if args.len() != 2 && args.len() != 3 {
        return Value::Null;
    }

    let in_features = match &args[0] {
        Value::Number(n) => {
            let count = *n as i64;
            if count <= 0 {
                return Value::Null;
            }
            count as usize
        }
        _ => return Value::Null,
    };

    let out_features = match &args[1] {
        Value::Number(n) => {
            let count = *n as i64;
            if count <= 0 {
                return Value::Null;
            }
            count as usize
        }
        _ => return Value::Null,
    };

    let use_bias = if args.len() == 3 {
        match &args[2] {
            Value::Number(n) => *n != 0.0,
            Value::Bool(b) => *b,
            _ => return Value::Null,
        }
    } else {
        true
    };

    match crate::layer::Linear::new(in_features, out_features, use_bias) {
        Ok(linear) => {
            let layer_id = crate::layer::add_layer_to_registry(Box::new(linear));
            crate::runtime::layer_value(layer_id)
        }
        Err(_) => Value::Null,
    }
}

/// Create a ReLU activation layer
/// relu_layer() -> layer_id
pub fn native_relu_layer(_args: &[Value]) -> Value {
    let relu = crate::layer::ReLU;
    let layer_id = crate::layer::add_layer_to_registry(Box::new(relu));
    crate::runtime::layer_value(layer_id)
}

/// Create a LeakyReLU layer: `leaky_relu_layer()` uses alpha=0.01; `leaky_relu_layer(alpha)` sets alpha.
pub fn native_leaky_relu_layer(args: &[Value]) -> Value {
    let alpha = match args.len() {
        0 => 0.01f32,
        1 => match &args[0] {
            Value::Number(n) => {
                let a = *n as f32;
                if a <= 0.0 {
                    return Value::Null;
                }
                a
            }
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::LeakyReLU::new(alpha);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

/// Create a Softmax activation layer: softmax_layer() default axis=1; softmax_layer(axis) with 0 or 1
pub fn native_softmax_layer(args: &[Value]) -> Value {
    let axis = match args.len() {
        0 => 1usize,
        1 => match &args[0] {
            Value::Number(n) => {
                let a = *n as i64;
                if a != 0 && a != 1 {
                    return Value::Null;
                }
                a as usize
            }
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let softmax = crate::layer::Softmax::with_axis(axis);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(softmax));
    crate::runtime::layer_value(layer_id)
}

/// Create a Flatten layer
/// flatten_layer() -> layer_id
pub fn native_flatten_layer(_args: &[Value]) -> Value {
    let flatten = crate::layer::Flatten;
    let layer_id = crate::layer::add_layer_to_registry(Box::new(flatten));
    crate::runtime::layer_value(layer_id)
}

/// Create a Sigmoid activation layer
/// sigmoid_layer() -> layer_id
pub fn native_sigmoid_layer(_args: &[Value]) -> Value {
    let sigmoid = crate::layer::Sigmoid::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(sigmoid));
    crate::runtime::layer_value(layer_id)
}

/// Create a Tanh activation layer
/// tanh_layer() -> layer_id
pub fn native_tanh_layer(_args: &[Value]) -> Value {
    let tanh = crate::layer::Tanh::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(tanh));
    crate::runtime::layer_value(layer_id)
}

pub fn native_log_softmax_layer(args: &[Value]) -> Value {
    let axis = match args.len() {
        0 => 1usize,
        1 => match &args[0] {
            Value::Number(n) => {
                let a = *n as i64;
                if a != 0 && a != 1 {
                    return Value::Null;
                }
                a as usize
            }
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::LogSoftmax::with_axis(axis);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_gelu_layer(_args: &[Value]) -> Value {
    let layer = crate::layer::Gelu::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_softplus_layer(_args: &[Value]) -> Value {
    let layer = crate::layer::Softplus::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

/// elu_layer(alpha) — default alpha=1.0
pub fn native_elu_layer(args: &[Value]) -> Value {
    let alpha = match args.len() {
        0 => 1.0f32,
        1 => match &args[0] {
            Value::Number(n) => *n as f32,
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::Elu::new(alpha);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_selu_layer(_args: &[Value]) -> Value {
    let layer = crate::layer::Selu::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

/// prelu_layer(init_alpha) — default 0.25
/// dropout_layer(p) — вероятность отключения (0..1)
pub fn native_dropout_layer(args: &[Value]) -> Value {
    let p = match args.len() {
        0 => 0.5f32,
        1 => match &args[0] {
            Value::Number(n) => {
                let v = *n as f32;
                if v < 0.0 || v >= 1.0 {
                    return Value::Null;
                }
                v
            }
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::Dropout::new(p);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_dropout2d_layer(args: &[Value]) -> Value {
    let p = match args.len() {
        0 => 0.5f32,
        1 => match &args[0] {
            Value::Number(n) => {
                let v = *n as f32;
                if v < 0.0 || v >= 1.0 {
                    return Value::Null;
                }
                v
            }
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::Dropout2d::new(p);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_dropconnect_layer(args: &[Value]) -> Value {
    let p = match args.len() {
        0 => 0.5f32,
        1 => match &args[0] {
            Value::Number(n) => {
                let v = *n as f32;
                if v < 0.0 || v >= 1.0 {
                    return Value::Null;
                }
                v
            }
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::DropConnect::new(p);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

/// conv2d_layer(in_c, out_c, kh, kw) — stride=1, padding=0, bias=true
/// max_pool2d_layer(kh, kw) — stride = kernel (неперекрывающийся)
pub fn native_max_pool2d_layer(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }
    let parse = |i: usize| -> Option<usize> {
        match &args[i] {
            Value::Number(n) => {
                let v = *n as i64;
                if v <= 0 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            _ => None,
        }
    };
    let kh = match parse(0) {
        Some(x) => x,
        None => return Value::Null,
    };
    let kw = match parse(1) {
        Some(x) => x,
        None => return Value::Null,
    };
    let layer = crate::layer::MaxPool2d::new(kh, kw, kh, kw);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_conv2d_layer(args: &[Value]) -> Value {
    if args.len() != 4 {
        return Value::Null;
    }
    let parse = |i: usize| -> Option<usize> {
        match &args[i] {
            Value::Number(n) => {
                let v = *n as i64;
                if v <= 0 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            _ => None,
        }
    };
    let in_c = match parse(0) {
        Some(x) => x,
        None => return Value::Null,
    };
    let out_c = match parse(1) {
        Some(x) => x,
        None => return Value::Null,
    };
    let kh = match parse(2) {
        Some(x) => x,
        None => return Value::Null,
    };
    let kw = match parse(3) {
        Some(x) => x,
        None => return Value::Null,
    };
    match crate::layer::Conv2d::new(in_c, out_c, (kh, kw), (1, 1), (0, 0), true) {
        Ok(c) => {
            let layer_id = crate::layer::add_layer_to_registry(Box::new(c));
            crate::runtime::layer_value(layer_id)
        }
        Err(_) => Value::Null,
    }
}

/// conv1d_layer(in_c, out_c, k) — stride 1, padding 0, bias on
pub fn native_conv1d_layer(args: &[Value]) -> Value {
    if args.len() != 3 {
        return Value::Null;
    }
    let parse = |i: usize| -> Option<usize> {
        match &args[i] {
            Value::Number(n) => {
                let v = *n as i64;
                if v <= 0 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            _ => None,
        }
    };
    let in_c = match parse(0) {
        Some(x) => x,
        None => return Value::Null,
    };
    let out_c = match parse(1) {
        Some(x) => x,
        None => return Value::Null,
    };
    let k = match parse(2) {
        Some(x) => x,
        None => return Value::Null,
    };
    match crate::layer::Conv1d::new(in_c, out_c, k, 1, 0, true) {
        Ok(c) => {
            let layer_id = crate::layer::add_layer_to_registry(Box::new(c));
            crate::runtime::layer_value(layer_id)
        }
        Err(_) => Value::Null,
    }
}

pub fn native_max_pool1d_layer(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }
    let parse = |i: usize| -> Option<usize> {
        match &args[i] {
            Value::Number(n) => {
                let v = *n as i64;
                if v <= 0 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            _ => None,
        }
    };
    let k = match parse(0) {
        Some(x) => x,
        None => return Value::Null,
    };
    let stride = match parse(1) {
        Some(x) => x,
        None => return Value::Null,
    };
    let layer = crate::layer::MaxPool1d::new(k, stride);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_avg_pool1d_layer(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }
    let parse = |i: usize| -> Option<usize> {
        match &args[i] {
            Value::Number(n) => {
                let v = *n as i64;
                if v <= 0 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            _ => None,
        }
    };
    let k = match parse(0) {
        Some(x) => x,
        None => return Value::Null,
    };
    let stride = match parse(1) {
        Some(x) => x,
        None => return Value::Null,
    };
    let layer = crate::layer::AvgPool1d::new(k, stride);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

/// avg_pool2d_layer(kh, kw, sy, sx)
pub fn native_avg_pool2d_layer(args: &[Value]) -> Value {
    if args.len() != 4 {
        return Value::Null;
    }
    let parse = |i: usize| -> Option<usize> {
        match &args[i] {
            Value::Number(n) => {
                let v = *n as i64;
                if v <= 0 {
                    None
                } else {
                    Some(v as usize)
                }
            }
            _ => None,
        }
    };
    let kh = match parse(0) {
        Some(x) => x,
        None => return Value::Null,
    };
    let kw = match parse(1) {
        Some(x) => x,
        None => return Value::Null,
    };
    let sy = match parse(2) {
        Some(x) => x,
        None => return Value::Null,
    };
    let sx = match parse(3) {
        Some(x) => x,
        None => return Value::Null,
    };
    let layer = crate::layer::AvgPool2d::new(kh, kw, sy, sx);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_global_max_pool_layer(args: &[Value]) -> Value {
    if !args.is_empty() {
        return Value::Null;
    }
    let layer = crate::layer::GlobalMaxPool2d::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

pub fn native_global_avg_pool_layer(args: &[Value]) -> Value {
    if !args.is_empty() {
        return Value::Null;
    }
    let layer = crate::layer::GlobalAvgPool2d::new();
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

fn native_placeholder_layer_impl(name: &'static str) -> Value {
    let layer = crate::layer::PlaceholderLayer::new(name);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

macro_rules! native_placeholder_layer_fn {
    ($name:ident, $lit:literal) => {
        pub fn $name(args: &[Value]) -> Value {
            let _ = args;
            native_placeholder_layer_impl($lit)
        }
    };
}

native_placeholder_layer_fn!(native_layer_conv3d_stub, "conv3d");
native_placeholder_layer_fn!(native_layer_depthwise_conv2d_stub, "depthwise_conv2d");
native_placeholder_layer_fn!(native_layer_separable_conv2d_stub, "separable_conv2d");
native_placeholder_layer_fn!(native_layer_transposed_conv2d_stub, "transposed_conv2d");
native_placeholder_layer_fn!(native_layer_batch_norm1d_stub, "batch_norm1d");
native_placeholder_layer_fn!(native_layer_batch_norm2d_stub, "batch_norm2d");
native_placeholder_layer_fn!(native_layer_layer_norm_stub, "layer_norm");
native_placeholder_layer_fn!(native_layer_instance_norm_stub, "instance_norm");
native_placeholder_layer_fn!(native_layer_group_norm_stub, "group_norm");
native_placeholder_layer_fn!(native_layer_rnn_stub, "rnn");
native_placeholder_layer_fn!(native_layer_lstm_stub, "lstm");
native_placeholder_layer_fn!(native_layer_gru_stub, "gru");
native_placeholder_layer_fn!(native_layer_attention_stub, "attention");
native_placeholder_layer_fn!(native_layer_self_attention_stub, "self_attention");
native_placeholder_layer_fn!(native_layer_multihead_attention_stub, "multihead_attention");
native_placeholder_layer_fn!(native_layer_embedding_stub, "embedding");
native_placeholder_layer_fn!(native_layer_positional_encoding_stub, "positional_encoding");
native_placeholder_layer_fn!(native_layer_reshape_stub, "reshape");
native_placeholder_layer_fn!(native_layer_permute_stub, "permute");
native_placeholder_layer_fn!(native_layer_concatenate_stub, "concatenate");
native_placeholder_layer_fn!(native_layer_stack_stub, "stack");
native_placeholder_layer_fn!(native_layer_add_stub, "add");
native_placeholder_layer_fn!(native_layer_residual_stub, "residual");
native_placeholder_layer_fn!(native_layer_skip_connection_stub, "skip_connection");
native_placeholder_layer_fn!(native_layer_upsample_stub, "upsample");
native_placeholder_layer_fn!(native_layer_upsample_nearest_stub, "upsample_nearest");
native_placeholder_layer_fn!(native_layer_upsample_bilinear_stub, "upsample_bilinear");
native_placeholder_layer_fn!(native_layer_graph_conv_stub, "graph_conv");
native_placeholder_layer_fn!(native_layer_graph_attention_stub, "graph_attention");
native_placeholder_layer_fn!(native_layer_transformer_block_stub, "transformer_block");
native_placeholder_layer_fn!(native_layer_feed_forward_stub, "feed_forward");

pub fn native_prelu_layer(args: &[Value]) -> Value {
    let init = match args.len() {
        0 => 0.25f32,
        1 => match &args[0] {
            Value::Number(n) => *n as f32,
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    let layer = crate::layer::PReLU::new(init);
    let layer_id = crate::layer::add_layer_to_registry(Box::new(layer));
    crate::runtime::layer_value(layer_id)
}

/// Call a layer with input tensor
/// layer_call(layer_id, input_tensor) -> output_tensor
pub fn native_layer_call(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let layer_id = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Layer as u8 => *id as crate::layer::LayerId,
        _ => return Value::Null,
    };

    let Some(input_tensor) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    // Convert Tensor to Variable (without requires_grad for inference)
    use crate::autograd::Variable;
    let input_var = Variable::new(input_tensor, false);
    
    // Call layer forward with Variable
    match crate::layer::forward_layer_var(layer_id, input_var) {
        Some(output_var) => {
            // Extract Tensor from Variable
            let output_tensor = output_var.data.borrow().clone();
            crate::runtime::tensor_to_value(output_tensor)
        }
        None => Value::Null,
    }
}

/// Invoke a bound method thunk: VM passes `[callee, receiver, ...method_args]`.
fn dispatch_nn_bound_invoke(args: &[Value]) -> Value {
    use crate::bound_method::{lookup_bound, lookup_nn_bound, BoundMethodPayload, NnBoundMethodKind};

    let bound_id = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::BoundMethod as u8 => *id,
        _ => return Value::Null,
    };

    if let Some(BoundMethodPayload::DatasetConcat { dataset_id }) = lookup_bound(bound_id) {
        if args.len() != 3 {
            return Value::Null;
        }
        match &args[1] {
            Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 && *id == dataset_id => {
                return native_dataset_concat(&args[1..]);
            }
            _ => return Value::Null,
        }
    }

    if let Some(BoundMethodPayload::DatasetPushData { dataset_id }) = lookup_bound(bound_id) {
        if args.len() != 4 {
            return Value::Null;
        }
        match &args[1] {
            Value::PluginOpaque { tag, id } if *tag == MlValueKind::Dataset as u8 && *id == dataset_id => {
                return native_dataset_push_data(&args[1..]);
            }
            _ => return Value::Null,
        }
    }

    if let Some(BoundMethodPayload::DatasetSplit { dataset_id }) = lookup_bound(bound_id) {
        use crate::native_error::set_native_error;
        if args.len() == 3 {
            match &args[1] {
                Value::PluginOpaque { tag, id } if *id == dataset_id => {
                    if *tag != MlValueKind::DatasetCatalog as u8 {
                        set_native_error(
                            "dataset.split(\"train\"|\"test\") is only for datasets from ml.load_dataset(...)"
                                .to_string(),
                        );
                        return Value::Null;
                    }
                    let kind = match crate::runtime::get_dataset_catalog_kind(*id) {
                        Some(k) => k,
                        None => return Value::Null,
                    };
                    let split = match &args[2] {
                        Value::String(s) => s.as_str(),
                        _ => {
                            set_native_error(
                                "dataset.split: use \"train\" or \"test\" as the first argument"
                                    .to_string(),
                            );
                            return Value::Null;
                        }
                    };
                    if split != "train" && split != "test" {
                        set_native_error(
                            "dataset.split: first argument must be \"train\" or \"test\"".to_string(),
                        );
                        return Value::Null;
                    }
                    match crate::builtin_datasets::materialize_catalog_split(kind, split) {
                        Ok(ds) => return crate::runtime::dataset_to_value(ds),
                        Err(e) => {
                            set_native_error(e);
                            return Value::Null;
                        }
                    }
                }
                _ => return Value::Null,
            }
        }
        if args.len() != 8 {
            return Value::Null;
        }
        match &args[1] {
            Value::PluginOpaque { tag, id } if *id == dataset_id => {
                if *tag == MlValueKind::DatasetCatalog as u8 {
                    let kind = match crate::runtime::get_dataset_catalog_kind(*id) {
                        Some(k) => k,
                        None => return Value::Null,
                    };
                    let full = match crate::builtin_datasets::materialize_catalog_full(kind) {
                        Ok(ds) => ds,
                        Err(e) => {
                            set_native_error(e);
                            return Value::Null;
                        }
                    };
                    return native_dataset_split_on_materialized(&full, &args[2..8]);
                }
                if *tag == MlValueKind::Dataset as u8 {
                    return native_dataset_split(&args[1..]);
                }
            }
            _ => {}
        }
        return Value::Null;
    }

    let Some((stored_nn_id, kind)) = lookup_nn_bound(bound_id) else {
        return Value::Null;
    };
    let recv = match &args[1] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => {
            if *id != stored_nn_id {
                return Value::Null;
            }
            args[1].clone()
        }
        _ => return Value::Null,
    };

    match kind {
        NnBoundMethodKind::Device => {
            if args.len() != 3 {
                return Value::Null;
            }
            native_nn_set_device(&[recv, args[2].clone()])
        }
        NnBoundMethodKind::GetDevice => {
            if args.len() != 2 {
                return Value::Null;
            }
            native_nn_get_device(std::slice::from_ref(&recv))
        }
        NnBoundMethodKind::Save => {
            if args.len() != 3 {
                return Value::Null;
            }
            native_nn_save(&[recv, args[2].clone()])
        }
        NnBoundMethodKind::Train => {
            // `native_nn_train` expects 7–10 args: nn + train args; VM adds bound + nn + … → 8–11 total.
            if args.len() < 8 || args.len() > 11 {
                return Value::Null;
            }
            native_nn_train(&args[1..])
        }
    }
}

/// Unified entry for callable plugin opaque values (VM dispatches here via `native_plugin_call`).
///
/// - **2 args** `[container, key_or_arg]`: `GetArrayElement`, layer call, forward, dataset fields, or
///   `NeuralNetwork` + string → bound-method thunk.
/// - **3+ args** `[callee, receiver, ...]`: method call after `GetArrayElement` returned a `BoundMethod`;
///   also `native_layer_call`-style stays 2-arg only on the first branch.
pub fn native_plugin_call(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }

    // Method invoke: `[bound_method, receiver, ...]`
    if let Value::PluginOpaque { tag, .. } = &args[0] {
        if *tag == MlValueKind::BoundMethod as u8 {
            return dispatch_nn_bound_invoke(args);
        }
    }

    if args.len() != 2 {
        return Value::Null;
    }

    let tag = match &args[0] {
        Value::PluginOpaque { tag, .. } => *tag,
        Value::Tensor(_) => MlValueKind::Tensor as u8,
        _ => return Value::Null,
    };
    // Tensor["shape"] / tensor.shape — GetArrayElement on PluginOpaque (tag Tensor) routes here.
    if tag == MlValueKind::Tensor as u8 {
        if let Value::String(s) = &args[1] {
            if s == "shape" {
                return native_shape(std::slice::from_ref(&args[0]));
            }
            if s == "data" || s == "to_array" {
                return native_data(std::slice::from_ref(&args[0]));
            }
            if s == "max" {
                return native_tensor_max(std::slice::from_ref(&args[0]));
            }
            if s == "min" {
                return native_tensor_min(std::slice::from_ref(&args[0]));
            }
            if s == "max_idx" {
                return native_max_idx(std::slice::from_ref(&args[0]));
            }
            if s == "min_idx" {
                return native_min_idx(std::slice::from_ref(&args[0]));
            }
            if s == "repr" {
                return native_tensor_repr(std::slice::from_ref(&args[0]));
            }
            if s == "T" || s == "transpose" {
                return native_transpose(std::slice::from_ref(&args[0]));
            }
            // Встроенные `sum` / `average` в VM дергают `native_plugin_call(_, "sum"|"mean")` (см. data-code array.rs).
            if s == "sum" {
                return native_sum(std::slice::from_ref(&args[0]));
            }
            if s == "mean" || s == "average" {
                return native_mean(std::slice::from_ref(&args[0]));
            }
        } else if let Value::Number(n) = &args[1] {
            // `t[i]` — срез по оси 0 (строка для 2D, элемент для 1D)
            let idx = *n as i64;
            if idx < 0 {
                return Value::Null;
            }
            let idx = idx as usize;
            let Some(t) = crate::runtime::tensor_data_clone(&args[0]) else {
                return Value::Null;
            };
            let cpu = match t.to_cpu() {
                Ok(t) => t,
                Err(_) => return Value::Null,
            };
            return match cpu.get_row(idx) {
                Ok(row) => crate::runtime::tensor_to_value(row),
                Err(_) => Value::Null,
            };
        }
        return Value::Null;
    }
    if tag == MlValueKind::Layer as u8 {
        native_layer_call(args)
    } else if tag == MlValueKind::NeuralNetwork as u8 {
        if crate::runtime::as_tensor_ref(&args[1]).is_some() {
            native_nn_forward(args)
        } else if let Value::String(name) = &args[1] {
            if let Some(k) = crate::bound_method::nn_method_kind_for_name(name.as_str()) {
                let nn_id = match &args[0] {
                    Value::PluginOpaque { id, .. } => *id,
                    _ => return Value::Null,
                };
                crate::bound_method::bound_method_value(nn_id, k)
            } else {
                Value::Null
            }
        } else {
            use crate::native_error::set_native_error;
            let got = match &args[1] {
                Value::Null => "null",
                Value::Number(_) => "number",
                Value::Bool(_) => "bool",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::ByteBuffer(_) => "byte_buffer",
                Value::Tensor(_) => "tensor(handle missing in runtime)",
                Value::PluginOpaque { tag: t, .. } => {
                    return {
                        set_native_error(format!(
                            "NeuralNetwork forward: second argument must be a Tensor, got PluginOpaque tag={} (tensor handle not in runtime?)",
                            t
                        ));
                        Value::Null
                    };
                }
                Value::Object(_) => "object",
                Value::ObjectPtr(_) => "object_ptr",
                Value::Path(_) => "path",
            };
            set_native_error(format!(
                "NeuralNetwork forward: second argument must be a Tensor, got {}",
                got
            ));
            Value::Null
        }
    } else if tag == MlValueKind::LinearRegression as u8 || tag == MlValueKind::Sequential as u8 {
        // Forward only when the second value is a tensor: model(tensor).
        if crate::runtime::as_tensor_ref(&args[1]).is_some() {
            native_nn_forward(args)
        } else {
            Value::Null
        }
    } else if tag == MlValueKind::Dataset as u8 {
        if let Value::String(s) = &args[1] {
            if s == "len" {
                return native_dataset_len(std::slice::from_ref(&args[0]));
            }
            if s == "features" {
                return native_dataset_features(std::slice::from_ref(&args[0]));
            }
            if s == "targets" {
                return native_dataset_targets(std::slice::from_ref(&args[0]));
            }
            if s == "split" {
                let ds_id = match &args[0] {
                    Value::PluginOpaque { id, .. } => *id,
                    _ => return Value::Null,
                };
                return crate::bound_method::bound_dataset_split_value(ds_id);
            }
            if s == "concat" {
                let ds_id = match &args[0] {
                    Value::PluginOpaque { id, .. } => *id,
                    _ => return Value::Null,
                };
                return crate::bound_method::bound_dataset_concat_value(ds_id);
            }
            if s == "push_data" {
                let ds_id = match &args[0] {
                    Value::PluginOpaque { id, .. } => *id,
                    _ => return Value::Null,
                };
                return crate::bound_method::bound_dataset_push_data_value(ds_id);
            }
        }
        native_dataset_get(args)
    } else if tag == MlValueKind::DatasetCatalog as u8 {
        use crate::native_error::set_native_error;
        if let Value::String(s) = &args[1] {
            if s == "split" {
                let ds_id = match &args[0] {
                    Value::PluginOpaque { id, .. } => *id,
                    _ => return Value::Null,
                };
                return crate::bound_method::bound_dataset_split_value(ds_id);
            }
            if matches!(s.as_str(), "len" | "features" | "targets" | "concat" | "push_data") {
                set_native_error(
                    "load_dataset: call .split(...) first to materialize the dataset".to_string(),
                );
                return Value::Null;
            }
        }
        set_native_error(
            "load_dataset: call .split(...) first before using the dataset".to_string(),
        );
        Value::Null
    } else {
        Value::Null
    }
}

/// Create a Sequential container
/// sequential(layers_array) -> sequential
/// layers_array: array of Layer values (Value::Layer)
pub fn native_sequential(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let layers_array = match &args[0] {
        Value::Array(arr) => {
            let arr_ref = arr.borrow();
            let mut layer_ids = Vec::new();
            for val in arr_ref.iter() {
                match val {
                    Value::PluginOpaque { tag, id } if *tag == MlValueKind::Layer as u8 => layer_ids.push(*id as crate::layer::LayerId),
                    _ => return Value::Null, // All elements must be Layer
                }
            }
            layer_ids
        }
        _ => return Value::Null,
    };

    // Convert layer_ids to Sequential
    // Create Sequential with layer_ids - layers will be accessed from registry when needed
    use crate::layer::Sequential;
    let seq = Sequential {
        layers: vec![],  // Empty - parameters() will get layers from registry via layer_ids
        layer_ids: layers_array,
    };
    crate::runtime::sequential_to_value(seq)
}

/// Add a layer to Sequential container
/// sequential_add(sequential, layer) -> null
pub fn native_sequential_add(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let sequential = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Sequential as u8 => crate::runtime::get_sequential(*id).expect("seq"),
        _ => return Value::Null,
    };

    let layer_id = match &args[1] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Layer as u8 => *id as crate::layer::LayerId,
        _ => return Value::Null,
    };

    sequential.borrow_mut().add(layer_id);
    Value::Null
}

/// Create a Neural Network from Sequential
/// neural_network(sequential) -> neural_network
pub fn native_neural_network(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let sequential = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Sequential as u8 => crate::runtime::get_sequential(*id).expect("seq"),
        _ => return Value::Null,
    };

    // Clone the Sequential (it now uses LayerId, so cloning is straightforward)
    let seq_clone = (*sequential.borrow()).clone();
    
    match NeuralNetwork::new(seq_clone) {
        Ok(nn) => crate::runtime::neural_network_to_value(nn),
        Err(_) => Value::Null,
    }
}

/// Forward pass through neural network
/// nn_forward(nn, x) -> predictions
/// Supports both NeuralNetwork and Sequential
pub fn native_nn_forward(args: &[Value]) -> Value {
    if args.len() != 2 {
        use crate::native_error::set_native_error;
        set_native_error("nn_forward requires exactly 2 arguments: (model, x)".to_string());
        return Value::Null;
    }

    let Some(x) = crate::runtime::tensor_data_clone(&args[1]) else {
        use crate::native_error::set_native_error;
        set_native_error("nn_forward: second argument must be a Tensor".to_string());
        return Value::Null;
    };

    // Support NeuralNetwork, Sequential, and LinearRegression
    match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => {
            let n = crate::runtime::get_neural_network(*id).expect("nn");
            let fwd = {
                let mut m = n.borrow_mut();
                m.forward(&x)
            };
            match fwd {
                Ok(output) => crate::runtime::tensor_to_value(output),
                Err(e) => {
                    use crate::native_error::set_native_error;
                    set_native_error(format!("NeuralNetwork forward failed: {}", e));
                    Value::Null
                }
            }
        }
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Sequential as u8 => {
            use crate::autograd::Variable;
            let s = crate::runtime::get_sequential(*id).expect("seq");
            let input_var = Variable::new(x.clone(), false);
            let output_var = s.borrow().forward(input_var);
            let output_tensor = output_var.data.borrow().clone();
            crate::runtime::tensor_to_value(output_tensor)
        }
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::LinearRegression as u8 => {
            let lr = crate::runtime::get_linear_regression(*id).expect("lr");
            let fwd = {
                let m = lr.borrow();
                m.forward(&x)
            };
            match fwd {
                Ok(output) => crate::runtime::tensor_to_value(output),
                Err(e) => {
                    use crate::native_error::set_native_error;
                    set_native_error(format!("LinearRegression forward failed: {}", e));
                    Value::Null
                }
            }
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_forward: first argument must be a NeuralNetwork, Sequential, or LinearRegression".to_string());
            Value::Null
        }
    }
}

/// Train neural network with early stopping and learning rate scheduling
/// nn_train_sh(nn, x, y, epochs, batch_size, lr, loss_type, optimizer, monitor, patience, min_delta, restore_best, x_val, y_val) -> training_history
pub fn native_nn_train_sh(args: &[Value]) -> Value {
    // Expected arguments: model, x_train, y_train, epochs, batch_size, learning_rate,
    // loss, optimizer, monitor, patience, min_delta, restore_best, x_val, y_val
    if args.len() < 12 || args.len() > 14 {
        use crate::native_error::set_native_error;
        set_native_error(format!("nn_train_sh requires 12-14 arguments, got {}", args.len()));
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: first argument must be a NeuralNetwork".to_string());
            return Value::Null;
        }
    };

    let Some(x) = crate::runtime::tensor_data_clone(&args[1]) else {
        use crate::native_error::set_native_error;
        set_native_error("nn_train_sh: x_train must be a Tensor".to_string());
        return Value::Null;
    };

    let Some(y) = crate::runtime::tensor_data_clone(&args[2]) else {
        use crate::native_error::set_native_error;
        set_native_error("nn_train_sh: y_train must be a Tensor".to_string());
        return Value::Null;
    };

    let epochs = match &args[3] {
        Value::Number(n) => {
            let e = *n as i64;
            if e <= 0 {
                use crate::native_error::set_native_error;
                set_native_error("nn_train_sh: epochs must be positive".to_string());
                return Value::Null;
            }
            e as usize
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: epochs must be a number".to_string());
            return Value::Null;
        }
    };

    let batch_size = match &args[4] {
        Value::Number(n) => {
            let b = *n as i64;
            if b <= 0 {
                use crate::native_error::set_native_error;
                set_native_error("nn_train_sh: batch_size must be positive".to_string());
                return Value::Null;
            }
            b as usize
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: batch_size must be a number".to_string());
            return Value::Null;
        }
    };

    let learning_rate = match &args[5] {
        Value::Number(n) => {
            let lr = *n as f64;
            if lr <= 0.0 {
                use crate::native_error::set_native_error;
                set_native_error("nn_train_sh: learning_rate must be positive".to_string());
                return Value::Null;
            }
            lr as f32
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: learning_rate must be a number".to_string());
            return Value::Null;
        }
    };

    let loss_type = match &args[6] {
        Value::String(s) => s.as_str(),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: loss must be a string".to_string());
            return Value::Null;
        }
    };

    let optimizer = match &args[7] {
        Value::String(s) => Some(s.as_str()),
        Value::Null => None,
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: optimizer must be a string or null".to_string());
            return Value::Null;
        }
    };

    let monitor = match &args[8] {
        Value::String(s) => s.as_str(),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: monitor must be a string".to_string());
            return Value::Null;
        }
    };

    let patience = match &args[9] {
        Value::Number(n) => {
            let p = *n as i64;
            if p <= 0 {
                use crate::native_error::set_native_error;
                set_native_error("nn_train_sh: patience must be positive".to_string());
                return Value::Null;
            }
            p as usize
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: patience must be a number".to_string());
            return Value::Null;
        }
    };

    let min_delta = match &args[10] {
        Value::Number(n) => {
            let d = *n as f64;
            if d < 0.0 {
                use crate::native_error::set_native_error;
                set_native_error("nn_train_sh: min_delta must be non-negative".to_string());
                return Value::Null;
            }
            d as f32
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: min_delta must be a number".to_string());
            return Value::Null;
        }
    };

    let restore_best = match &args[11] {
        Value::Bool(b) => *b,
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_train_sh: restore_best must be a boolean".to_string());
            return Value::Null;
        }
    };

    // Extract validation data if provided (optional, args[12] and args[13])
    let (x_val, y_val) = if args.len() >= 14 {
        match (
            crate::runtime::tensor_data_clone(&args[12]),
            crate::runtime::tensor_data_clone(&args[13]),
        ) {
            (Some(xa), Some(ya)) => (Some(xa), Some(ya)),
            _ => (None, None),
        }
    } else {
        (None, None)
    };

    // Call train_sh
    let result = nn.borrow_mut().train_sh(
        &x, &y,
        epochs, batch_size, learning_rate,
        loss_type,
        optimizer,
        monitor,
        patience,
        min_delta,
        restore_best,
        x_val.as_ref(),
        y_val.as_ref(),
    );

    match result {
        Ok(history) => {
            // Convert TrainingHistorySH to DataCode Value (Object/Dict)
            use std::collections::HashMap;
            let mut obj = HashMap::new();
            
            // Convert loss array
            let loss_values: Vec<Value> = history.loss.iter().map(|&v| Value::Number(v as f64)).collect();
            obj.insert("loss".to_string(), Value::Array(Rc::new(RefCell::new(loss_values))));
            
            // Convert val_loss array (optional)
            if let Some(ref val_loss) = history.val_loss {
                let val_loss_values: Vec<Value> = val_loss.iter().map(|&v| Value::Number(v as f64)).collect();
                obj.insert("val_loss".to_string(), Value::Array(Rc::new(RefCell::new(val_loss_values))));
            } else {
                obj.insert("val_loss".to_string(), Value::Null);
            }
            
            // Convert acc array
            let acc_values: Vec<Value> = history.acc.iter().map(|&v| Value::Number(v as f64)).collect();
            obj.insert("acc".to_string(), Value::Array(Rc::new(RefCell::new(acc_values))));
            
            // Convert val_acc array (optional)
            if let Some(ref val_acc) = history.val_acc {
                let val_acc_values: Vec<Value> = val_acc.iter().map(|&v| Value::Number(v as f64)).collect();
                obj.insert("val_acc".to_string(), Value::Array(Rc::new(RefCell::new(val_acc_values))));
            } else {
                obj.insert("val_acc".to_string(), Value::Null);
            }
            
            // Convert lr array
            let lr_values: Vec<Value> = history.lr.iter().map(|&v| Value::Number(v as f64)).collect();
            obj.insert("lr".to_string(), Value::Array(Rc::new(RefCell::new(lr_values))));
            
            // Convert scalar fields
            obj.insert("best_metric".to_string(), Value::Number(history.best_metric as f64));
            obj.insert("best_epoch".to_string(), Value::Number(history.best_epoch as f64));
            obj.insert("stopped_epoch".to_string(), Value::Number(history.stopped_epoch as f64));
            
            Value::Object(Rc::new(RefCell::new(obj)))
        }
        Err(e) => {
            use crate::native_error::set_native_error;
            set_native_error(format!("Training failed: {}", e));
            Value::Null
        }
    }
}

/// Train neural network
/// nn_train(nn, x, y, epochs, batch_size, lr, loss_type) -> loss_history (array of floats, ABI-safe)
/// nn_train(nn, x, y, epochs, batch_size, lr, loss_type, x_val, y_val) -> loss_history
pub fn native_nn_train(args: &[Value]) -> Value {
    // Support both old API (7 or 9 args) and new API with optimizer (8 or 10 args)
    if args.len() < 7 || args.len() > 10 {
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => return Value::Null,
    };

    let Some(x) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    let Some(y) = crate::runtime::tensor_data_clone(&args[2]) else {
        return Value::Null;
    };

    let epochs = match &args[3] {
        Value::Number(n) => {
            let e = *n as i64;
            if e <= 0 {
                return Value::Null;
            }
            e as usize
        }
        _ => return Value::Null,
    };

    let batch_size = match &args[4] {
        Value::Number(n) => {
            let b = *n as i64;
            if b <= 0 {
                return Value::Null;
            }
            b as usize
        }
        _ => return Value::Null,
    };

    let lr = match &args[5] {
        Value::Number(n) => {
            let rate = *n as f64;
            if rate <= 0.0 {
                return Value::Null;
            }
            rate as f32
        }
        _ => return Value::Null,
    };

    let loss_type = match &args[6] {
        Value::String(s) => s.as_str(),
        _ => return Value::Null,
    };

    // Extract optimizer if provided (args[7] for new API, or None for old API)
    let optimizer = if args.len() >= 8 {
        match &args[7] {
            Value::String(s) => Some(s.as_str()),
            _ => None, // If not a string, treat as old API
        }
    } else {
        None
    };

    // Extract validation data if provided
    // Since resolve_function_args only includes provided arguments, we need to check
    // if validation data is present at the end of the args array
    // Base arguments: nn (0), x (1), y (2), epochs (3), batch_size (4), lr (5), loss_type (6)
    // With optimizer: optimizer is at position 7
    // Validation data: x_val and y_val are at positions 8 and 9 if present
    let (x_val, y_val) = if args.len() >= 9 {
        // Check if last two arguments are Tensors (validation data)
        let last_idx = args.len() - 1;
        let second_last_idx = args.len() - 2;
        match (
            crate::runtime::tensor_data_clone(&args[second_last_idx]),
            crate::runtime::tensor_data_clone(&args[last_idx]),
        ) {
            (Some(xa), Some(ya)) => (Some(xa), Some(ya)),
            _ => (None, None),
        }
    } else {
        (None, None)
    };

    // Call train with validation data if available
    let result = if let (Some(x_val_tensor), Some(y_val_tensor)) = (x_val, y_val) {
        // Validation data provided
        nn.borrow_mut().train(&x, &y, epochs, batch_size, lr, loss_type, Some(&x_val_tensor), Some(&y_val_tensor), optimizer)
    } else {
        // No validation data
        nn.borrow_mut().train(&x, &y, epochs, batch_size, lr, loss_type, None, None, optimizer)
    };

    match result {
        Ok((loss_history, _accuracy_history, _val_loss_history, _val_accuracy_history)) => {
            // Return loss series as a plain array so the value round-trips through the VM ABI bridge.
            // (Value::Object cannot be round-tripped: object handles are scoped to the plugin bridge.)
            let loss_values: Vec<Value> = loss_history.iter().map(|&v| Value::Number(v as f64)).collect();
            Value::Array(Rc::new(RefCell::new(loss_values)))
        }
        Err(e) => {
            // Set error message for VM to handle
            use crate::native_error::set_native_error;
            set_native_error(format!("Training failed: {}", e));
            Value::Null
        }
    }
}

/// Save neural network to file
/// nn_save(nn, path) -> null
pub fn native_nn_save(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => return Value::Null,
    };

    let file_path = match &args[1] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => return Value::Null,
    };

    let path = file_path.to_string_lossy().to_string();

    let save_res = {
        let b = nn.borrow();
        b.save(&path)
    };
    match save_res {
        Ok(_) => Value::Null,
        Err(_) => Value::Null,
    }
}

/// Load neural network from file
/// nn_load(path) -> neural_network
pub fn native_nn_load(args: &[Value]) -> Value {
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error("nn_load() requires exactly 1 argument: path".to_string());
        return Value::Null;
    }

    let file_path = match &args[0] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("nn_load() requires a string or path argument".to_string());
            return Value::Null;
        }
    };

    let path = file_path.to_string_lossy().to_string();

    match NeuralNetwork::load(&path) {
        Ok(nn) => crate::runtime::neural_network_to_value(nn),
        Err(e) => {
            use crate::native_error::set_native_error;
            set_native_error(format!("Failed to load model from '{}': {}", path, e));
            Value::Null
        }
    }
}

/// Save model to file (ML module function)
/// ml.save_model(model, path) -> null
pub fn native_ml_save_model(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => return Value::Null,
    };

    let file_path = match &args[1] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => return Value::Null,
    };

    let path = file_path.to_string_lossy().to_string();

    let save_res = {
        let b = nn.borrow();
        b.save(&path)
    };
    match save_res {
        Ok(_) => Value::Null,
        Err(_) => Value::Null,
    }
}

/// Load model from file (ML module function)
/// ml.load(path) -> neural_network
pub fn native_ml_load_model(args: &[Value]) -> Value {
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error("ml.load() requires exactly 1 argument: path".to_string());
        return Value::Null;
    }

    let file_path = match &args[0] {
        Value::Path(p) => p.clone(),
        Value::String(s) => PathBuf::from(s),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("ml.load() requires a string or path argument".to_string());
            return Value::Null;
        }
    };

    let path = file_path.to_string_lossy().to_string();

    match NeuralNetwork::load(&path) {
        Ok(nn) => crate::runtime::neural_network_to_value(nn),
        Err(e) => {
            use crate::native_error::set_native_error;
            set_native_error(format!("Failed to load model from '{}': {}", path, e));
            Value::Null
        }
    }
}

/// load_dataset("mnist" | "cifar-10" | "cifar-100") -> dataset catalog (materialize with `.split(...)`).
pub fn native_load_dataset(args: &[Value]) -> Value {
    use crate::datasets::parse_builtin_dataset_name;
    use crate::native_error::set_native_error;

    if args.len() != 1 {
        set_native_error(
            "ml.load_dataset expects one string: \"mnist\", \"cifar-10\", or \"cifar-100\"".to_string(),
        );
        return Value::Null;
    }

    let name = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => {
            set_native_error("ml.load_dataset: argument must be a string".to_string());
            return Value::Null;
        }
    };

    let Some(kind) = parse_builtin_dataset_name(name) else {
        set_native_error(format!(
            "ml.load_dataset: unknown dataset name {:?} (expected mnist, cifar-10, cifar-100)",
            name
        ));
        return Value::Null;
    };

    if let Err(e) = crate::builtin_datasets::ensure_builtin_dataset_ready(kind) {
        set_native_error(e);
        return Value::Null;
    }

    crate::runtime::dataset_catalog_to_value(kind)
}

/// Load MNIST dataset
/// load_mnist("train") or load_mnist("test") -> dataset
pub fn native_load_mnist(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;

    if args.len() != 1 {
        set_native_error("ml.load_mnist expects exactly one string argument: \"train\" or \"test\"".to_string());
        return Value::Null;
    }

    let split = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => {
            set_native_error("ml.load_mnist: argument must be a string".to_string());
            return Value::Null;
        }
    };

    let (images_path, labels_path) = match crate::datasets_manager::resolve_mnist_paths(split) {
        Ok(p) => p,
        Err(e) => {
            set_native_error(e);
            return Value::Null;
        }
    };

    match Dataset::from_mnist(&images_path, &labels_path) {
        Ok(dataset) => crate::runtime::dataset_to_value(dataset),
        Err(e) => {
            set_native_error(format!("load_mnist: {}", e));
            Value::Null
        }
    }
}

/// Categorical cross entropy loss (one-hot targets)
/// categorical_cross_entropy_loss(logits, targets) -> loss_tensor
pub fn native_categorical_cross_entropy_loss(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }

    let Some(logits) = crate::runtime::tensor_data_clone(&args[0]) else {
        return Value::Null;
    };

    let Some(targets) = crate::runtime::tensor_data_clone(&args[1]) else {
        return Value::Null;
    };

    match categorical_cross_entropy_loss(&logits, &targets) {
        Ok(loss) => crate::runtime::tensor_to_value(loss),
        Err(_) => Value::Null,
    }
}

fn ml_set_global_device(device: Device) -> Value {
    GLOBAL_ML_DEVICE.with(|global_device| {
        *global_device.borrow_mut() = Some(device.clone());
    });
    Value::String(device.name().to_string())
}

/// Fallback to CPU with a unified warning (requested name, detail, registry list).
fn ml_fallback_cpu_with_warning(
    requested: &str,
    detail: impl std::fmt::Display,
    registry: &crate::backend_registry::BackendRegistry,
) -> Value {
    use crate::native_error::set_native_error;
    let avail = registry.available_devices().join(", ");
    let msg = format!(
        "Device '{}' unavailable ({}). Available: [{}]. Falling back to CPU.",
        requested,
        detail,
        avail
    );
    set_native_error(msg);
    ml_set_global_device(Device::Cpu)
}

/// Set default device for ML operations
/// ml_set_device("cpu"|"cuda"|"metal"|"auto"|"gpu") — `gpu` and `auto` pick the best backend for this OS (see BackendRegistry::auto_select).
/// Returns device name as string on success, null on unknown device.
/// Falls back to CPU with a warning if a GPU backend was requested but is not compiled in or not available at runtime.
pub fn native_ml_set_device(args: &[Value]) -> Value {
    use crate::backend_registry::BackendRegistry;
    use crate::native_error::set_native_error;

    if args.len() != 1 {
        return Value::Null;
    }

    let device_str = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Value::Null,
    };

    let registry = BackendRegistry::detect();

    // "auto" and "gpu" → same policy as Device::default / BackendRegistry::auto_select (Metal on macOS, CUDA on Linux/Windows).
    let resolved: &str = if device_str == "auto" || device_str == "gpu" {
        registry.auto_select()
    } else {
        device_str
    };

    match resolved {
        "cpu" => ml_set_global_device(Device::Cpu),
        "metal" => {
            if !registry.metal {
                return ml_fallback_cpu_with_warning(
                    device_str,
                    "Metal not compiled in or no compatible GPU at runtime",
                    &registry,
                );
            }
            match Device::from_str("metal") {
                Ok(device) => ml_set_global_device(device),
                Err(e) => ml_fallback_cpu_with_warning(device_str, e, &registry),
            }
        }
        "cuda" => {
            if !registry.cuda {
                return ml_fallback_cpu_with_warning(
                    device_str,
                    "CUDA not compiled in or no compatible GPU at runtime",
                    &registry,
                );
            }
            match Device::from_str("cuda") {
                Ok(device) => ml_set_global_device(device),
                Err(e) => ml_fallback_cpu_with_warning(device_str, e, &registry),
            }
        }
        _ => match Device::from_str(device_str) {
            Ok(device) => ml_set_global_device(device),
            Err(e) => {
                set_native_error(format!("Failed to set device '{}': {}", device_str, e));
                Value::Null
            }
        },
    }
}

/// Get current default device
/// ml_get_device() -> device_name
pub fn native_ml_get_device(_args: &[Value]) -> Value {
    GLOBAL_ML_DEVICE.with(|global_device| {
        let device = global_device.borrow();
        let device = device.as_ref()
            .cloned()
            .unwrap_or_else(Device::default);
        Value::String(device.name().to_string())
    })
}

/// Set device for neural network
/// nn_set_device(nn, device_str) -> device_name
/// Uses BackendRegistry for two-level checking (compile-time + runtime)
pub fn native_nn_set_device(args: &[Value]) -> Value {
    use crate::backend_registry::BackendRegistry;
    
    if args.len() != 2 {
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => return Value::Null,
    };

    let device_str = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Value::Null,
    };

    let registry = BackendRegistry::detect();

    // "auto" and "gpu" → same policy as ml_set_device / BackendRegistry::auto_select
    let device_str = if device_str == "auto" || device_str == "gpu" {
        registry.auto_select()
    } else {
        device_str
    };

    let avail_list = || registry.available_devices().join(", ");

    // Two-level checking: registry first, then Device::from_str
    match device_str {
        "cpu" => {
            let cpu_device = Device::Cpu;
            nn.borrow_mut().set_device(cpu_device.clone());
            Value::String(cpu_device.name().to_string())
        }
        "metal" => {
            if !registry.metal {
                let error_msg = format!(
                    "Device 'metal' unavailable (Metal not compiled in or no GPU at runtime). Available: [{}]",
                    avail_list()
                );
                use crate::native_error::set_native_error;
                set_native_error(error_msg.clone());
                eprintln!("{}", error_msg);
                return Value::Null;
            }

            #[cfg(feature = "metal")]
            {
                match Device::from_str("metal") {
                    Ok(device) => {
                        nn.borrow_mut().set_device(device.clone());
                        Value::String(device.name().to_string())
                    }
                    Err(e) => {
                        let error_msg = format!(
                            "Device 'metal' unavailable ({}). Available: [{}]",
                            e,
                            avail_list()
                        );
                        use crate::native_error::set_native_error;
                        set_native_error(error_msg.clone());
                        eprintln!("{}", error_msg);
                        Value::Null
                    }
                }
            }

            #[cfg(not(feature = "metal"))]
            {
                let error_msg = format!(
                    "Device 'metal' unavailable (Metal not compiled into this build). Available: [{}]",
                    avail_list()
                );
                use crate::native_error::set_native_error;
                set_native_error(error_msg.clone());
                eprintln!("{}", error_msg);
                Value::Null
            }
        }
        "cuda" => {
            if !registry.cuda {
                let error_msg = format!(
                    "Device 'cuda' unavailable (CUDA not compiled in or no GPU at runtime). Available: [{}]",
                    avail_list()
                );
                use crate::native_error::set_native_error;
                set_native_error(error_msg.clone());
                eprintln!("{}", error_msg);
                return Value::Null;
            }

            #[cfg(feature = "cuda")]
            {
                match Device::from_str("cuda") {
                    Ok(device) => {
                        nn.borrow_mut().set_device(device.clone());
                        Value::String(device.name().to_string())
                    }
                    Err(e) => {
                        let error_msg = format!(
                            "Device 'cuda' unavailable ({}). Available: [{}]",
                            e,
                            avail_list()
                        );
                        use crate::native_error::set_native_error;
                        set_native_error(error_msg.clone());
                        eprintln!("{}", error_msg);
                        Value::Null
                    }
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                let error_msg = format!(
                    "Device 'cuda' unavailable (CUDA not compiled into this build). Available: [{}]",
                    avail_list()
                );
                use crate::native_error::set_native_error;
                set_native_error(error_msg.clone());
                eprintln!("{}", error_msg);
                Value::Null
            }
        }
        _ => {
            let error_msg = format!(
                "Unknown device '{}'. Available: [{}]",
                device_str,
                avail_list()
            );
            use crate::native_error::set_native_error;
            set_native_error(error_msg.clone());
            eprintln!("{}", error_msg);
            Value::Null
        }
    }
}

/// Get device for neural network
/// nn_get_device(nn) -> device_name
pub fn native_nn_get_device(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => return Value::Null,
    };

    let device = nn.borrow().get_device();
    Value::String(device.name().to_string())
}

/// Get list of available backends (compile-time features + runtime probe).
/// Same as `ml.available_backends()` — `["cpu", ...]` with optional `"metal"` / `"cuda"`.
pub fn native_devices(_args: &[Value]) -> Value {
    use crate::backend_registry::BackendRegistry;
    use std::rc::Rc;
    use std::cell::RefCell;
    
    let registry = BackendRegistry::detect();
    let devices = registry.available_devices();
    Value::Array(Rc::new(RefCell::new(
        devices.into_iter().map(|s| Value::String(s.to_string())).collect()
    )))
}

/// Validate neural network model
/// ml.validate_model(model) -> bool
/// Returns true if model is valid (not null, has layers, can be used)
pub fn native_ml_validate_model(args: &[Value]) -> Value {
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error("ml.validate_model() requires exactly 1 argument: model".to_string());
        return Value::Bool(false);
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        Value::Null => {
            use crate::native_error::set_native_error;
            set_native_error("Model is null - cannot validate null model".to_string());
            return Value::Bool(false);
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("ml.validate_model() requires a NeuralNetwork object as argument".to_string());
            return Value::Bool(false);
        }
    };

    let nn_ref = nn.borrow();
    let sequential = nn_ref.sequential();
    
    // Check if Sequential has layers
    let layer_ids = sequential.layer_ids();
    if layer_ids.is_empty() {
        use crate::native_error::set_native_error;
        set_native_error("Model validation failed: Sequential container has no layers".to_string());
        return Value::Bool(false);
    }

    // Check if sequential has layers
    if sequential.layers.is_empty() {
        use crate::native_error::set_native_error;
        set_native_error("Model validation failed: Sequential has no layers".to_string());
        return Value::Bool(false);
    }

    // Model appears valid
    Value::Bool(true)
}

/// Print model information
/// Format optimizer string with parameters
fn format_optimizer_string(optimizer_type: &str, optimizer_params: &Option<serde_json::Value>) -> String {
    if let Some(ref params) = optimizer_params {
        if let Some(lr) = params.get("lr").and_then(|v| v.as_f64()) {
            match optimizer_type {
                "SGD" => format!("SGD(lr={})", lr),
                "Momentum" => {
                    if let Some(beta) = params.get("beta").and_then(|v| v.as_f64()) {
                        format!("Momentum(lr={}, beta={})", lr, beta)
                    } else {
                        format!("Momentum(lr={})", lr)
                    }
                },
                "NAG" => {
                    if let Some(beta) = params.get("beta").and_then(|v| v.as_f64()) {
                        format!("NAG(lr={}, beta={})", lr, beta)
                    } else {
                        format!("NAG(lr={})", lr)
                    }
                },
                "Adagrad" => {
                    if let Some(eps) = params.get("epsilon").and_then(|v| v.as_f64()) {
                        format!("Adagrad(lr={}, epsilon={})", lr, eps)
                    } else {
                        format!("Adagrad(lr={})", lr)
                    }
                },
                "RMSprop" => {
                    let gamma = params.get("gamma").and_then(|v| v.as_f64()).unwrap_or(0.99);
                    let eps = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);
                    format!("RMSprop(lr={}, gamma={}, epsilon={})", lr, gamma, eps)
                },
                "Adam" => {
                    let beta1 = params.get("beta1").and_then(|v| v.as_f64()).unwrap_or(0.9);
                    let beta2 = params.get("beta2").and_then(|v| v.as_f64()).unwrap_or(0.999);
                    let eps = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);
                    format!("Adam(lr={}, beta1={}, beta2={}, epsilon={})", lr, beta1, beta2, eps)
                },
                "AdamW" => {
                    let beta1 = params.get("beta1").and_then(|v| v.as_f64()).unwrap_or(0.9);
                    let beta2 = params.get("beta2").and_then(|v| v.as_f64()).unwrap_or(0.999);
                    let eps = params.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);
                    let wd = params.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.01);
                    format!("AdamW(lr={}, beta1={}, beta2={}, epsilon={}, weight_decay={})", lr, beta1, beta2, eps, wd)
                },
                _ => format!("{}(lr={})", optimizer_type, lr),
            }
        } else {
            optimizer_type.to_string()
        }
    } else {
        optimizer_type.to_string()
    }
}

/// ml.model_info(model, verbose=false, format="text", show_graph=false)
/// Returns null for text format, JSON string for json format
pub fn native_ml_model_info(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    use crate::layer::with_layer;
    use serde_json;
    
    if args.is_empty() {
        set_native_error("ml.model_info() requires at least 1 argument: model".to_string());
        return Value::Null;
    }

    // Parse model (required argument)
    let model_value = &args[0];
    
    // Parse optional arguments
    let mut verbose = false;
    let mut format = "text".to_string();
    let mut show_graph = false;
    
    // Parse optional arguments by type and position
    for i in 1..args.len() {
        match &args[i] {
            Value::Bool(b) => {
                // Could be verbose or show_graph - use position to determine
                if i == 1 {
                    verbose = *b;
                } else {
                    show_graph = *b;
                }
            }
            Value::String(s) => {
                // Must be format
                format = s.clone();
            }
            _ => {
                // Ignore unknown types
            }
        }
    }
    
    // Validate format
    if format != "text" && format != "json" {
        format = "text".to_string();
    }

    // Handle NeuralNetwork
    if let Value::PluginOpaque { tag, id } = model_value {
        if *tag == MlValueKind::NeuralNetwork as u8 {
        let nn = crate::runtime::get_neural_network(*id).expect("nn");
        let nn_ref = nn.borrow();
        let sequential = nn_ref.sequential();
        let device = nn_ref.get_device();
        
        // Get layer information
        let layer_ids = sequential.layer_ids();
        if layer_ids.is_empty() {
            set_native_error("Model has no layers".to_string());
            return Value::Null;
        }
        
        let mut layers_info = Vec::new();
        let mut total_params = 0;
        let mut input_size = 0;
        let mut current_size = 0;
        
        // Process each layer
        for (idx, &layer_id) in layer_ids.iter().enumerate() {
            let layer_info = with_layer(layer_id, |layer| {
                let params = layer.parameters();
                let layer_type = format!("{:?}", layer);
                let layer_type_clean = if layer_type.contains("Linear") {
                    "Linear"
                } else if layer_type.contains("PlaceholderLayer") {
                    "Placeholder"
                } else if layer_type.contains("GlobalMaxPool2d") {
                    "GlobalMaxPool2d"
                } else if layer_type.contains("GlobalAvgPool2d") {
                    "GlobalAvgPool2d"
                } else if layer_type.contains("AvgPool2d") {
                    "AvgPool2d"
                } else if layer_type.contains("AvgPool1d") {
                    "AvgPool1d"
                } else if layer_type.contains("MaxPool1d") {
                    "MaxPool1d"
                } else if layer_type.contains("MaxPool2d") {
                    "MaxPool2d"
                } else if layer_type.contains("Conv1d") {
                    "Conv1d"
                } else if layer_type.contains("Conv2d") {
                    "Conv2d"
                } else if layer_type.contains("LeakyReLU") {
                    "LeakyReLU"
                } else if layer_type.contains("PReLU") {
                    "PReLU"
                } else if layer_type.contains("ReLU") {
                    "ReLU"
                } else if layer_type.contains("LogSoftmax") {
                    "LogSoftmax"
                } else if layer_type.contains("Softmax") {
                    "Softmax"
                } else if layer_type.contains("Gelu") {
                    "GELU"
                } else if layer_type.contains("Softplus") {
                    "Softplus"
                } else if layer_type.contains("Selu") {
                    "SELU"
                } else if layer_type.contains("Elu") {
                    "ELU"
                } else if layer_type.contains("Dropout2d") {
                    "Dropout2d"
                } else if layer_type.contains("DropConnect") {
                    "DropConnect"
                } else if layer_type.contains("Dropout") {
                    "Dropout"
                } else if layer_type.contains("Sigmoid") {
                    "Sigmoid"
                } else if layer_type.contains("Tanh") {
                    "Tanh"
                } else if layer_type.contains("Flatten") {
                    "Flatten"
                } else {
                    "Unknown"
                };
                
                let in_features = layer.in_features();
                let out_features = layer.out_features();
                let is_trainable = layer.is_trainable();
                
                // Calculate parameters for this layer
                let layer_params = if params.len() >= 2 {
                    // Linear layer: weight + bias (from initialized parameters)
                    let weight_params = if let Some((_, weight_tensor)) = params.get(0) {
                        weight_tensor.shape.iter().product()
                    } else {
                        0
                    };
                    let bias_params = if let Some((_, bias_tensor)) = params.get(1) {
                        bias_tensor.shape.iter().product()
                    } else {
                        0
                    };
                    weight_params + bias_params
                } else if params.is_empty() && layer_type_clean == "Linear" {
                    // Linear layer with uninitialized parameters (e.g., loaded model)
                    // Calculate from layer dimensions: weights (in_features * out_features) + bias (out_features)
                    in_features * out_features + out_features
                } else {
                    0
                };
                
                (layer_type_clean.to_string(), in_features, out_features, layer_params, is_trainable)
            });
            
            if let Some((layer_type, in_feat, out_feat, layer_params, is_trainable)) = layer_info {
                total_params += layer_params;
                
                // Determine actual input/output sizes
                // For Linear layers, use their explicit sizes
                // For activation layers (in_feat == 0, out_feat == 0), use current_size
                let actual_in = if in_feat > 0 {
                    if idx == 0 {
                        input_size = in_feat;
                    }
                    in_feat
                } else {
                    // Activation layer - use current size from previous layer
                    current_size
                };
                
                let actual_out = if out_feat > 0 {
                    out_feat
                } else {
                    // Activation layer preserves input size
                    actual_in
                };
                
                // Update current_size for next layer
                if actual_out > 0 {
                    current_size = actual_out;
                }
                
                layers_info.push((layer_type, actual_in, actual_out, layer_params, is_trainable));
            }
        }
        
        // Get trainable and frozen parameter counts from the model
        let (trainable_params, frozen_params) = nn_ref.count_trainable_frozen_params();
        let frozen_layers = nn_ref.get_frozen_layers();
        
        // Build computation graph string if requested
        let graph_str = if show_graph {
            let mut graph_parts = Vec::new();
            graph_parts.push("Input".to_string());
            for (layer_type, _, _, _, _) in &layers_info {
                graph_parts.push(layer_type.clone());
            }
            graph_parts.join(" → ")
        } else {
            String::new()
        };
        
        // Determine model type
        let model_type = if layers_info.len() > 2 {
            "MLP"
        } else {
            "NeuralNetwork"
        };
        
        // Output based on format
        if format == "json" {
            // Build JSON structure
            let mut layers_json = Vec::new();
            for (layer_type, in_feat, out_feat, params, is_trainable) in &layers_info {
                let mut layer_obj = serde_json::json!({
                    "type": layer_type,
                    "in": in_feat,
                    "out": out_feat,
                    "params": params
                });
                if verbose {
                    layer_obj["trainable"] = if *params > 0 {
                        serde_json::Value::Bool(*is_trainable)
                    } else {
                        serde_json::Value::Null
                    };
                }
                layers_json.push(layer_obj);
            }
            
            let mut json_obj = serde_json::json!({
                "type": model_type,
                "device": device.name(),
                "dtype": "f32",
                "parameters": {
                    "trainable": trainable_params,
                    "frozen": frozen_params,
                    "total": total_params
                },
                "layers": layers_json
            });
            
            // Add frozen layers list if any
            if !frozen_layers.is_empty() {
                json_obj["frozen_layers"] = serde_json::json!(frozen_layers);
            }
            
            if show_graph && !graph_str.is_empty() {
                json_obj["graph"] = serde_json::Value::String(graph_str);
            }
            
            // Add training information and parameter statistics if verbose
            if verbose {
                let training_stages = nn_ref.training_stages();
                if !training_stages.is_empty() {
                    let mut stages_json = Vec::new();
                    for stage in training_stages {
                        let mut stage_obj = serde_json::json!({
                            "epochs": stage.epochs,
                            "loss": stage.loss,
                            "optimizer_type": stage.optimizer_type,
                            "frozen_layers": stage.frozen_layers,
                            "trainable_params": stage.trainable_params,
                            "frozen_params": stage.frozen_params,
                            "loss_history": stage.loss_history,
                            "accuracy_history": stage.accuracy_history
                        });
                        if let Some(ref opt_params) = stage.optimizer_params {
                            stage_obj["optimizer_params"] = opt_params.clone();
                        }
                        if let Some(ref val_loss) = stage.val_loss_history {
                            stage_obj["val_loss_history"] = serde_json::json!(val_loss);
                        }
                        if let Some(ref val_acc) = stage.val_accuracy_history {
                            stage_obj["val_accuracy_history"] = serde_json::json!(val_acc);
                        }
                        if let Some(ref lr_history) = stage.lr_history {
                            stage_obj["lr_history"] = serde_json::json!(lr_history);
                        }
                        stages_json.push(stage_obj);
                    }
                    json_obj["training"] = serde_json::json!({
                        "total_epochs": nn_ref.total_epochs(),
                        "stages": stages_json
                    });
                } else {
                    // Add legacy training info if available
                    if let Some(epochs) = nn_ref.training_epochs() {
                        let mut training_obj = serde_json::json!({
                            "total_epochs": epochs
                        });
                        if let Some(loss) = nn_ref.training_loss() {
                            training_obj["loss"] = serde_json::json!(loss);
                        }
                        if let Some(optimizer) = nn_ref.training_optimizer() {
                            training_obj["optimizer"] = serde_json::json!(optimizer);
                        }
                        if let Some((loss_history, acc_history)) = nn_ref.training_history() {
                            training_obj["loss_history"] = serde_json::json!(loss_history);
                            training_obj["accuracy_history"] = serde_json::json!(acc_history);
                        }
                        if let Some((val_loss, val_acc)) = nn_ref.validation_history() {
                            training_obj["val_loss_history"] = serde_json::json!(val_loss);
                            training_obj["val_accuracy_history"] = serde_json::json!(val_acc);
                        }
                        json_obj["training"] = training_obj;
                    }
                }
                
                // Add parameter statistics
                let mut param_stats = Vec::new();
                let mut layer_idx = 0;
                for &layer_id in layer_ids.iter() {
                    let layer_stats = with_layer(layer_id, |layer| {
                        let params = layer.parameters_var();
                        if params.len() >= 2 {
                            // Linear layer with weights and bias
                            // Get weight tensor from Variable
                            let weight_tensor = params[0].data.borrow().clone();
                            
                            // Get bias tensor from Variable
                            let bias_tensor = params[1].data.borrow().clone();
                            
                            let weight_cpu = weight_tensor.to_cpu().unwrap_or_else(|_| weight_tensor);
                            let bias_cpu = bias_tensor.to_cpu().unwrap_or_else(|_| bias_tensor);
                            
                            let weight_min = weight_cpu.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let weight_max = weight_cpu.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let weight_mean = weight_cpu.as_slice().iter().sum::<f32>() / weight_cpu.as_slice().len() as f32;
                            
                            let bias_min = bias_cpu.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let bias_max = bias_cpu.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let bias_mean = bias_cpu.as_slice().iter().sum::<f32>() / bias_cpu.as_slice().len() as f32;
                            
                            Some((layer_idx, weight_min, weight_max, weight_mean, weight_cpu.as_slice().len(), bias_min, bias_max, bias_mean, bias_cpu.as_slice().len()))
                        } else {
                            None
                        }
                    });
                    
                    if let Some(Some((idx, w_min, w_max, w_mean, w_count, b_min, b_max, b_mean, b_count))) = layer_stats {
                        param_stats.push(serde_json::json!({
                            "layer": idx,
                            "weights": {
                                "min": w_min,
                                "max": w_max,
                                "mean": w_mean,
                                "count": w_count
                            },
                            "bias": {
                                "min": b_min,
                                "max": b_max,
                                "mean": b_mean,
                                "count": b_count
                            }
                        }));
                        layer_idx += 1;
                    }
                }
                if !param_stats.is_empty() {
                    json_obj["parameter_statistics"] = serde_json::json!(param_stats);
                }
            }
            
            let json_str = serde_json::to_string_pretty(&json_obj)
                .unwrap_or_else(|_| "{}".to_string());
            // CString in the ABI bridge rejects interior NUL bytes
            let json_str = json_str.replace('\0', "");
            
            // Print to stdout
            println!("{}", json_str);
            
            Value::String(json_str)
        } else {
            // Text format output
            println!("Model type: {}", model_type);
            println!("Device: {}", device.name());
            println!("DType: f32");
            println!();
            println!("Architecture:");
            println!("--------------------------------------------------------------------------");
            
            // Print header with or without Trainable column based on verbose
            if verbose {
                println!("{:<20} {:<20} {:<15} {}", "Layer (type)", "Input → Output", "Params", "Trainable");
            } else {
                println!("{:<20} {:<20} {}", "Layer (type)", "Input → Output", "Params");
            }
            println!("--------------------------------------------------------------------------");
            
            // Print all layers (including input if we know the size)
            if input_size > 0 && !layers_info.is_empty() {
                // Check if first layer is not Input
                if let Some((first_type, _, _, _, _)) = layers_info.first() {
                    if first_type != "Input" {
                        if verbose {
                            println!("{:<20} {:<20} {:<15} {}", "Input", format!("{} → {}", input_size, input_size), 0, "N/A");
                        } else {
                            println!("{:<20} {:<20} {}", "Input", format!("{} → {}", input_size, input_size), 0);
                        }
                    }
                }
            }
            
            // Print all layers
            for (layer_type, in_feat, out_feat, params, is_trainable) in &layers_info {
                if verbose {
                    let trainable_str = if *params > 0 {
                        if *is_trainable { "Yes" } else { "No" }
                    } else {
                        "N/A"
                    };
                    println!("{:<20} {:<20} {:<15} {}", layer_type, format!("{} → {}", in_feat, out_feat), params, trainable_str);
                } else {
                    println!("{:<20} {:<20} {}", layer_type, format!("{} → {}", in_feat, out_feat), params);
                }
            }
            
            println!("--------------------------------------------------------------------------");
            println!();
            println!("Trainable parameters: {}", trainable_params);
            println!("Frozen parameters: {}", frozen_params);
            println!("Total parameters: {}", total_params);
            
            // Print frozen layers if any
            if !frozen_layers.is_empty() {
                println!();
                println!("Frozen layers:");
                for layer_name in &frozen_layers {
                    println!("  - {}", layer_name);
                }
            }
            
            if show_graph && !graph_str.is_empty() {
                println!();
                println!("Computation graph:");
                println!("{}", graph_str);
            }
            
            // Print training information if verbose
            if verbose {
                let training_stages = nn_ref.training_stages();
                if !training_stages.is_empty() {
                    println!();
                    println!("Training Information:");
                    println!("--------------------------------------------------------------------------");
                    println!("Total epochs trained: {}", nn_ref.total_epochs());
                    println!("Training stages: {}", training_stages.len());
                    println!();
                    
                    for (idx, stage) in training_stages.iter().enumerate() {
                        println!("Stage {}:", idx + 1);
                        println!("  Epochs: {}", stage.epochs);
                        println!("  Loss: {}", stage.loss);
                        
                        let optimizer_str = format_optimizer_string(&stage.optimizer_type, &stage.optimizer_params);
                        println!("  Optimizer: {}", optimizer_str);
                        
                        // Print frozen layers
                        if stage.frozen_layers.is_empty() {
                            println!("  Frozen layers: none");
                        } else {
                            println!("  Frozen layers:");
                            for layer in &stage.frozen_layers {
                                println!("    - {}", layer);
                            }
                        }
                        
                        // Print parameters
                        println!("  Parameters:");
                        println!("    Trainable: {}", stage.trainable_params);
                        println!("    Frozen:    {}", stage.frozen_params);
                        
                        // Print full history (not truncated)
                        if !stage.loss_history.is_empty() {
                            let loss_str = stage.loss_history.iter()
                                .map(|v| format!("{:.4}", v))
                                .collect::<Vec<_>>()
                                .join(", ");
                            println!("  Loss history: [{}] ({} values)", loss_str, stage.loss_history.len());
                        }
                        
                        if !stage.accuracy_history.is_empty() {
                            let acc_str = stage.accuracy_history.iter()
                                .map(|v| format!("{:.2}%", v * 100.0))
                                .collect::<Vec<_>>()
                                .join(", ");
                            println!("  Accuracy history: [{}] ({} values)", acc_str, stage.accuracy_history.len());
                        }
                        
                        if let Some(ref val_loss) = stage.val_loss_history {
                            if !val_loss.is_empty() {
                                let val_loss_str = val_loss.iter()
                                    .map(|v| format!("{:.4}", v))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("  Val Loss: [{}] ({} values)", val_loss_str, val_loss.len());
                            }
                        }
                        
                        if let Some(ref val_acc) = stage.val_accuracy_history {
                            if !val_acc.is_empty() {
                                let val_acc_str = val_acc.iter()
                                    .map(|v| format!("{:.2}%", v * 100.0))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("  Val Accuracy: [{}] ({} values)", val_acc_str, val_acc.len());
                            }
                        }
                        
                        if let Some(ref lr_history) = stage.lr_history {
                            if !lr_history.is_empty() {
                                let lr_str = lr_history.iter()
                                    .map(|v| format!("{:.6}", v))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("  Learning Rate: [{}] ({} values)", lr_str, lr_history.len());
                            }
                        }
                        
                        println!();
                    }
                    
                    // Check if frozen layers changed between stages
                    if training_stages.len() > 1 {
                        let last_frozen = &training_stages[training_stages.len() - 1].frozen_layers;
                        let prev_frozen = &training_stages[training_stages.len() - 2].frozen_layers;
                        if last_frozen != prev_frozen {
                            println!("Info: frozen configuration changed since last training stage");
                        }
                    }
                } else {
                    // Check for legacy training info
                    if let Some(epochs) = nn_ref.training_epochs() {
                        println!();
                        println!("Training Information:");
                        println!("--------------------------------------------------------------------------");
                        println!("Total epochs trained: {}", epochs);
                        if let Some(loss) = nn_ref.training_loss() {
                            println!("Loss: {}", loss);
                        }
                        if let Some(optimizer) = nn_ref.training_optimizer() {
                            println!("Optimizer: {}", optimizer);
                        }
                        if let Some((loss_history, acc_history)) = nn_ref.training_history() {
                            if !loss_history.is_empty() {
                                let loss_str = loss_history.iter()
                                    .map(|v| format!("{:.4}", v))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("Loss history: [{}] ({} values)", loss_str, loss_history.len());
                            }
                            if !acc_history.is_empty() {
                                let acc_str = acc_history.iter()
                                    .map(|v| format!("{:.2}%", v * 100.0))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("Accuracy history: [{}] ({} values)", acc_str, acc_history.len());
                            }
                        }
                        if let Some((val_loss, val_acc)) = nn_ref.validation_history() {
                            if !val_loss.is_empty() {
                                let val_loss_str = val_loss.iter()
                                    .map(|v| format!("{:.4}", v))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("Val Loss: [{}] ({} values)", val_loss_str, val_loss.len());
                            }
                            if !val_acc.is_empty() {
                                let val_acc_str = val_acc.iter()
                                    .map(|v| format!("{:.2}%", v * 100.0))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                println!("Val Accuracy: [{}] ({} values)", val_acc_str, val_acc.len());
                            }
                        }
                    }
                }
                
                // Add parameter statistics if verbose
                println!();
                println!("Parameter Statistics:");
                println!("--------------------------------------------------------------------------");
                let mut layer_idx = 0;
                for &layer_id in layer_ids.iter() {
                    let layer_stats = with_layer(layer_id, |layer| {
                        let params = layer.parameters_var();
                        if params.len() >= 2 {
                            // Linear layer with weights and bias
                            // Get weight tensor from Variable
                            let weight_tensor = params[0].data.borrow().clone();
                            
                            // Get bias tensor from Variable
                            let bias_tensor = params[1].data.borrow().clone();
                            
                            let weight_cpu = weight_tensor.to_cpu().unwrap_or_else(|_| weight_tensor);
                            let bias_cpu = bias_tensor.to_cpu().unwrap_or_else(|_| bias_tensor);
                            
                            let weight_data = weight_cpu.data();
                            let bias_data = bias_cpu.data();
                            
                            let weight_min = weight_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let weight_max = weight_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let weight_mean = weight_data.iter().sum::<f32>() / weight_data.len() as f32;
                            let weight_count = weight_data.len();
                            
                            let bias_min = bias_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let bias_max = bias_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let bias_mean = bias_data.iter().sum::<f32>() / bias_data.len() as f32;
                            let bias_count = bias_data.len();
                            
                            Some((layer_idx, weight_min, weight_max, weight_mean, weight_count, bias_min, bias_max, bias_mean, bias_count))
                        } else {
                            None
                        }
                    });
                    
                    if let Some(Some((idx, w_min, w_max, w_mean, w_count, b_min, b_max, b_mean, b_count))) = layer_stats {
                        println!("Layer {} (Linear):", idx);
                        println!("  Weights:");
                        println!("    Min:   {:.6}", w_min);
                        println!("    Max:   {:.6}", w_max);
                        println!("    Mean:  {:.6}", w_mean);
                        println!("    Count: {}", w_count);
                        println!("  Bias:");
                        println!("    Min:   {:.6}", b_min);
                        println!("    Max:   {:.6}", b_max);
                        println!("    Mean:  {:.6}", b_mean);
                        println!("    Count: {}", b_count);
                        println!();
                        layer_idx += 1;
                    }
                }
            }
            
            Value::Null
        }
        } else if *tag == MlValueKind::LinearRegression as u8 {
        let lr = crate::runtime::get_linear_regression(*id).expect("lr");
        let lr_ref = lr.borrow();
        let weights = lr_ref.get_weights();
        let bias = lr_ref.get_bias();
        
        let feature_count = weights.shape[0];
        let weight_params = weights.shape.iter().product::<usize>();
        let bias_params = bias.shape.iter().product::<usize>();
        let total_params = weight_params + bias_params;
        
        if format == "json" {
            let mut json_obj = serde_json::json!({
                "type": "LinearRegression",
                "device": "CPU",
                "dtype": "f32",
                "parameters": {
                    "trainable": total_params,
                    "frozen": 0,
                    "total": total_params
                },
                "layers": [
                    {
                        "type": "Linear",
                        "in": feature_count,
                        "out": 1,
                        "params": total_params,
                        "trainable": true
                    }
                ]
            });
            
            // Add parameter statistics if verbose
            if verbose {
                // Calculate statistics for weights
                let weights_cpu = weights.to_cpu().unwrap_or_else(|_| weights.clone());
                let weight_min = weights_cpu.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let weight_max = weights_cpu.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weight_mean = weights_cpu.as_slice().iter().sum::<f32>() / weights_cpu.as_slice().len() as f32;
                
                // Calculate statistics for bias
                let bias_cpu = bias.to_cpu().unwrap_or_else(|_| bias.clone());
                let bias_min = bias_cpu.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let bias_max = bias_cpu.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let bias_mean = bias_cpu.as_slice().iter().sum::<f32>() / bias_cpu.as_slice().len() as f32;
                
                json_obj["parameter_statistics"] = serde_json::json!({
                    "weights": {
                        "min": weight_min,
                        "max": weight_max,
                        "mean": weight_mean,
                        "count": weight_params
                    },
                    "bias": {
                        "min": bias_min,
                        "max": bias_max,
                        "mean": bias_mean,
                        "count": bias_params
                    }
                });
            }
            
            let json_str = serde_json::to_string_pretty(&json_obj)
                .unwrap_or_else(|_| "{}".to_string());
            let json_str = json_str.replace('\0', "");
            
            println!("{}", json_str);
            Value::String(json_str)
        } else {
            println!("Model type: LinearRegression");
            println!("Device: CPU");
            println!("DType: f32");
            println!();
            println!("Architecture:");
            println!("-------------------------------------------------");
            if verbose {
                println!("{:<20} {:<20} {:<15} {}", "Layer (type)", "Input → Output", "Params", "Trainable");
            } else {
                println!("{:<20} {:<20} {}", "Layer (type)", "Input → Output", "Params");
            }
            println!("-------------------------------------------------");
            if verbose {
                println!("{:<20} {:<20} {:<15} {}", "Input", format!("{} → {}", feature_count, feature_count), 0, "N/A");
                println!("{:<20} {:<20} {:<15} {}", "Linear", format!("{} → 1", feature_count), total_params, "Yes");
            } else {
                println!("{:<20} {:<20} {}", "Input", format!("{} → {}", feature_count, feature_count), 0);
                println!("{:<20} {:<20} {}", "Linear", format!("{} → 1", feature_count), total_params);
            }
            println!("-------------------------------------------------");
            println!();
            println!("Trainable parameters: {}", total_params);
            println!("Frozen parameters: 0");
            println!("Total parameters: {}", total_params);
            
            // Add parameter statistics if verbose
            if verbose {
                println!();
                println!("Parameter Statistics:");
                println!("-------------------------------------------------");
                
                // Calculate statistics for weights
                let weights_cpu = weights.to_cpu().unwrap_or_else(|_| weights.clone());
                let weight_min = weights_cpu.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let weight_max = weights_cpu.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weight_mean = weights_cpu.as_slice().iter().sum::<f32>() / weights_cpu.as_slice().len() as f32;
                
                // Calculate statistics for bias
                let bias_cpu = bias.to_cpu().unwrap_or_else(|_| bias.clone());
                let bias_min = bias_cpu.as_slice().iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let bias_max = bias_cpu.as_slice().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let bias_mean = bias_cpu.as_slice().iter().sum::<f32>() / bias_cpu.as_slice().len() as f32;
                
                println!("Weights:");
                println!("  Min:   {:.6}", weight_min);
                println!("  Max:   {:.6}", weight_max);
                println!("  Mean:  {:.6}", weight_mean);
                println!("  Count: {}", weight_params);
                println!();
                println!("Bias:");
                println!("  Min:   {:.6}", bias_min);
                println!("  Max:   {:.6}", bias_max);
                println!("  Mean:  {:.6}", bias_mean);
                println!("  Count: {}", bias_params);
            }
            
            Value::Null
        }
        } else {
        set_native_error("ml.model_info() requires a NeuralNetwork or LinearRegression model as first argument".to_string());
        Value::Null
        }
    } else {
        set_native_error("ml.model_info() requires an ML model as first argument".to_string());
        Value::Null
    }
}

/// Get layer by index from a neural network model
/// model_get_layer(model, index) -> layer
pub fn native_model_get_layer(args: &[Value]) -> Value {
    if args.len() != 2 {
        use crate::native_error::set_native_error;
        set_native_error("model_get_layer requires exactly 2 arguments: (model, index)".to_string());
        return Value::Null;
    }

    let nn = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::NeuralNetwork as u8 => crate::runtime::get_neural_network(*id).expect("nn"),
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("model_get_layer: first argument must be a NeuralNetwork".to_string());
            return Value::Null;
        }
    };

    let index = match &args[1] {
        Value::Number(n) => {
            let idx = *n as i64;
            if idx < 0 {
                use crate::native_error::set_native_error;
                set_native_error("model_get_layer: index must be non-negative".to_string());
                return Value::Null;
            }
            idx as usize
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("model_get_layer: second argument must be a number".to_string());
            return Value::Null;
        }
    };

    let layer_opt = {
        let b = nn.borrow();
        b.get_layer(index)
    };
    match layer_opt {
        Some(layer_id) => crate::runtime::layer_value(layer_id),
        None => {
            use crate::native_error::set_native_error;
            set_native_error(format!("model_get_layer: layer index {} is out of bounds", index));
            Value::Null
        }
    }
}

/// Freeze a layer (disable parameter updates during training)
/// layer_freeze(layer) -> null
pub fn native_layer_freeze(args: &[Value]) -> Value {
    use crate::layer::with_layer;
    
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error("layer_freeze requires exactly 1 argument: layer".to_string());
        return Value::Null;
    }

    let layer_id = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Layer as u8 => *id as crate::layer::LayerId,
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("layer_freeze: argument must be a Layer".to_string());
            return Value::Null;
        }
    };

    // Call freeze through the Layer trait
    with_layer(layer_id, |layer| {
        layer.freeze();
    });

    Value::Null
}

/// Unfreeze a layer (enable parameter updates during training)
/// layer_unfreeze(layer) -> null
pub fn native_layer_unfreeze(args: &[Value]) -> Value {
    use crate::layer::with_layer;
    
    if args.len() != 1 {
        use crate::native_error::set_native_error;
        set_native_error("layer_unfreeze requires exactly 1 argument: layer".to_string());
        return Value::Null;
    }

    let layer_id = match &args[0] {
        Value::PluginOpaque { tag, id } if *tag == MlValueKind::Layer as u8 => *id as crate::layer::LayerId,
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("layer_unfreeze: argument must be a Layer".to_string());
            return Value::Null;
        }
    };

    // Call unfreeze through the Layer trait
    with_layer(layer_id, |layer| {
        layer.unfreeze();
    });

    Value::Null
}

