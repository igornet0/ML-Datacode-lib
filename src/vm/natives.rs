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

/// Helper function to recursively extract data and infer shape from nested arrays
/// Returns Ok((data, shape)) on success, or Err(error_message) on error
fn extract_tensor_data_and_shape(value: &Value) -> Result<(Vec<f32>, Vec<usize>), String> {
    match value {
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
            
            // Nested array - recursively process each element
            let mut all_shapes: Vec<Vec<usize>> = Vec::new();
            let mut all_data: Vec<Vec<f32>> = Vec::new();
            
            for (idx, val) in arr_ref.iter().enumerate() {
                match extract_tensor_data_and_shape(val) {
                    Ok((data, shape)) => {
                        all_data.push(data);
                        all_shapes.push(shape);
                    }
                    Err(msg) => {
                        return Err(format!("Error at row index {}: {}", idx, msg));
                    }
                }
            }
            
            if all_shapes.is_empty() {
                return Err("Empty nested array cannot be converted to tensor".to_string());
            }
            
            // Verify all sub-arrays have the same shape (check for ragged arrays)
            let first_shape = &all_shapes[0];
            let expected_row_length = if !first_shape.is_empty() {
                first_shape[0]
            } else {
                0
            };
            
            for (idx, shape) in all_shapes.iter().enumerate().skip(1) {
                if shape != first_shape {
                    // More detailed error for ragged arrays
                    let actual_row_length = if !shape.is_empty() {
                        shape[0]
                    } else {
                        0
                    };
                    return Err(format!(
                        "ShapeError: expected row length {}, but got {} at row index {}",
                        expected_row_length, actual_row_length, idx
                    ));
                }
            }
            
            // Flatten all data into single vector
            let mut flat_data = Vec::new();
            for data in all_data {
                flat_data.extend(data);
            }
            
            // Construct shape: [len(arr_ref), ...first_shape]
            let mut tensor_shape = vec![arr_ref.len()];
            tensor_shape.extend(first_shape);
            
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

/// Get tensor data as array
/// data(tensor)
pub fn native_data(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let data = tensor.borrow().data.clone();
    let data_values: Vec<Value> = data.iter().map(|&d| Value::Number(d as f64)).collect();
    Value::Array(Rc::new(RefCell::new(data_values)))
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

/// Sum all elements
/// sum(tensor)
pub fn native_sum(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    Value::Number(t.sum() as f64)
}

/// Mean of all elements
/// mean(tensor)
pub fn native_mean(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let Some(tensor) = crate::runtime::as_tensor_ref(&args[0]) else {
        return Value::Null;
    };

    let t = tensor.borrow().clone();
    Value::Number(t.mean() as f64)
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

/// Create a dataset from a table
/// dataset(table, feature_columns, target_columns) -> dataset
/// `table` is `AbiValue::Table` bridged to `Value::Array([headers, rows])`.
pub fn native_dataset(args: &[Value]) -> Value {
    if args.len() != 3 {
        use crate::native_error::set_native_error;
        set_native_error(format!("ml.dataset expects 3 arguments (table, feature_cols, target_cols), got {}", args.len()));
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
            set_native_error("ml.dataset: second argument (feature_cols) must be an array of strings".to_string());
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
                        set_native_error("ml.dataset: target_cols must be an array of strings".to_string());
                        return Value::Null;
                    }
                }
            }
            columns
        }
        _ => {
            use crate::native_error::set_native_error;
            set_native_error("ml.dataset: third argument (target_cols) must be an array of strings".to_string());
            return Value::Null;
        }
    };

    match Dataset::from_abi_table(&headers, &rows, &feature_columns_array, &target_columns_array) {
        Ok(dataset) => crate::runtime::dataset_to_value(dataset),
        Err(e) => {
            use crate::native_error::set_native_error;
            set_native_error(format!("ml.dataset failed: {}", e));
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

/// dataset_from_tensors(features_tensor, targets_tensor) -> dataset
/// Use this when `import ml` is a dylib: `Table` cannot be passed through the ABI.
pub fn native_dataset_from_tensors(args: &[Value]) -> Value {
    use crate::native_error::set_native_error;
    if args.len() != 2 {
        set_native_error(format!(
            "ml.dataset_from_tensors expects 2 arguments (features, targets), got {}",
            args.len()
        ));
        return Value::Null;
    }

    let Some(ft) = crate::runtime::tensor_data_clone(&args[0]) else {
        set_native_error("ml.dataset_from_tensors: features must be a Tensor".to_string());
        return Value::Null;
    };
    let Some(tt) = crate::runtime::tensor_data_clone(&args[1]) else {
        set_native_error("ml.dataset_from_tensors: targets must be a Tensor".to_string());
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

/// Convert integer labels to one-hot encoding
/// onehot(labels, num_classes?) -> onehot_tensor
/// If num_classes is not provided, it is determined as max(label) + 1
pub fn native_onehot(args: &[Value]) -> Value {
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
        labels_cpu.data.iter().map(|&x| x as usize).collect()
    } else if labels_cpu.ndim() == 2 && labels_cpu.shape[1] == 1 {
        // Shape [N, 1]
        labels_cpu.data.iter().map(|&x| x as usize).collect()
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

/// Create a Linear layer
/// linear_layer(in_features, out_features) -> layer_id
pub fn native_linear_layer(args: &[Value]) -> Value {
    if args.len() != 2 {
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

    match crate::layer::Linear::new(in_features, out_features, true) {
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

/// Create a Softmax activation layer
/// softmax_layer() -> layer_id
pub fn native_softmax_layer(_args: &[Value]) -> Value {
    let softmax = crate::layer::Softmax;
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

/// Unified entry for callable plugin opaque values (VM dispatches here via `native_plugin_call`).
pub fn native_plugin_call(args: &[Value]) -> Value {
    if args.len() != 2 {
        return Value::Null;
    }
    let tag = match &args[0] {
        Value::PluginOpaque { tag, .. } => *tag,
        _ => return Value::Null,
    };
    if tag == MlValueKind::Layer as u8 {
        native_layer_call(args)
    } else if tag == MlValueKind::NeuralNetwork as u8
        || tag == MlValueKind::LinearRegression as u8
        || tag == MlValueKind::Sequential as u8
    {
        native_nn_forward(args)
    } else if tag == MlValueKind::Dataset as u8 {
        native_dataset_get(args)
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

/// Load MNIST dataset
/// load_mnist("train") or load_mnist("test") -> dataset
pub fn native_load_mnist(args: &[Value]) -> Value {
    if args.len() != 1 {
        return Value::Null;
    }

    let split = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Value::Null,
    };

    // Determine file paths based on split
    // Try multiple possible locations for MNIST files
    let (images_file, labels_file) = match split {
        "train" => (
            crate::mnist_paths::TRAIN_IMAGES,
            crate::mnist_paths::TRAIN_LABELS,
        ),
        "test" => (
            crate::mnist_paths::T10K_IMAGES,
            crate::mnist_paths::T10K_LABELS,
        ),
        _ => return Value::Null,
    };

    // Try to resolve paths - try multiple possible locations
    use std::path::PathBuf;
    use std::env;
    
    // Helper function to find file in multiple possible locations
    fn find_file(filename: &str) -> Option<String> {
        let possible_paths = vec![
            filename.to_string(), // Original path
            format!("../{}", filename), // One level up
            format!("../../{}", filename), // Two levels up
            format!("../../../{}", filename), // Three levels up
        ];
        
        // Try relative to current directory
        if let Ok(cwd) = env::current_dir() {
            for path in &possible_paths {
                let full_path = cwd.join(path);
                if full_path.exists() {
                    return Some(full_path.to_string_lossy().to_string());
                }
            }
        }
        
        // Try direct path (absolute or relative)
        let direct = PathBuf::from(filename);
        if direct.exists() {
            return Some(direct.to_string_lossy().to_string());
        }
        
        // Try with CARGO_MANIFEST_DIR if available (when running from cargo)
        if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
            let manifest_path = PathBuf::from(manifest_dir).join(filename);
            if manifest_path.exists() {
                return Some(manifest_path.to_string_lossy().to_string());
            }
        }
        
        None
    }
    
    // Find images file
    let images_path = find_file(images_file);
    
    // Find labels file (in same directory as images if found, otherwise search)
    let labels_path = if let Some(ref img_path) = images_path {
        // If we found images, construct labels path in the same directory
        let mut path = PathBuf::from(img_path);
        path.set_file_name(labels_file.split('/').last().unwrap_or(labels_file));
        if path.exists() {
            path.to_string_lossy().to_string()
        } else {
            // Fallback to searching
            find_file(labels_file).unwrap_or_else(|| labels_file.to_string())
        }
    } else {
        // Search for labels file
        find_file(labels_file).unwrap_or_else(|| labels_file.to_string())
    };

    if let Some(ref img_path) = images_path {
        match Dataset::from_mnist(img_path, &labels_path) {
            Ok(dataset) => crate::runtime::dataset_to_value(dataset),
            Err(_e) => Value::Null,
        }
    } else {
        Value::Null
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
                } else if layer_type.contains("ReLU") {
                    "ReLU"
                } else if layer_type.contains("Sigmoid") {
                    "Sigmoid"
                } else if layer_type.contains("Tanh") {
                    "Tanh"
                } else if layer_type.contains("Softmax") {
                    "Softmax"
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
                            
                            let weight_min = weight_cpu.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let weight_max = weight_cpu.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let weight_mean = weight_cpu.data.iter().sum::<f32>() / weight_cpu.data.len() as f32;
                            
                            let bias_min = bias_cpu.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let bias_max = bias_cpu.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let bias_mean = bias_cpu.data.iter().sum::<f32>() / bias_cpu.data.len() as f32;
                            
                            Some((layer_idx, weight_min, weight_max, weight_mean, weight_cpu.data.len(), bias_min, bias_max, bias_mean, bias_cpu.data.len()))
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
                let weight_min = weights_cpu.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let weight_max = weights_cpu.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weight_mean = weights_cpu.data.iter().sum::<f32>() / weights_cpu.data.len() as f32;
                
                // Calculate statistics for bias
                let bias_cpu = bias.to_cpu().unwrap_or_else(|_| bias.clone());
                let bias_min = bias_cpu.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let bias_max = bias_cpu.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let bias_mean = bias_cpu.data.iter().sum::<f32>() / bias_cpu.data.len() as f32;
                
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
                let weight_min = weights_cpu.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let weight_max = weights_cpu.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weight_mean = weights_cpu.data.iter().sum::<f32>() / weights_cpu.data.len() as f32;
                
                // Calculate statistics for bias
                let bias_cpu = bias.to_cpu().unwrap_or_else(|_| bias.clone());
                let bias_min = bias_cpu.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let bias_max = bias_cpu.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let bias_mean = bias_cpu.data.iter().sum::<f32>() / bias_cpu.data.len() as f32;
                
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

