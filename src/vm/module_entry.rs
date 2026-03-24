//! ABI exports for `libml` — registered via `datacode_module` callback.

use datacode_abi::AbiValue;

pub fn shim_tensor(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_tensor, args)
}

pub fn shim_shape(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_shape, args)
}

pub fn shim_data(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_data, args)
}

pub fn shim_add(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_add, args)
}

pub fn shim_sub(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sub, args)
}

pub fn shim_mul(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_mul, args)
}

pub fn shim_matmul(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_matmul, args)
}

pub fn shim_transpose(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_transpose, args)
}

pub fn shim_sum(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sum, args)
}

pub fn shim_mean(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_mean, args)
}

pub fn shim_max_idx(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_max_idx, args)
}

pub fn shim_min_idx(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_min_idx, args)
}

pub fn shim_graph(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph, args)
}

pub fn shim_graph_add_input(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_add_input, args)
}

pub fn shim_graph_add_op(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_add_op, args)
}

pub fn shim_graph_forward(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_forward, args)
}

pub fn shim_graph_get_output(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_get_output, args)
}

pub fn shim_graph_backward(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_backward, args)
}

pub fn shim_graph_get_gradient(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_get_gradient, args)
}

pub fn shim_graph_zero_grad(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_zero_grad, args)
}

pub fn shim_graph_set_requires_grad(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_graph_set_requires_grad, args)
}

pub fn shim_linear_regression(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_linear_regression, args)
}

pub fn shim_lr_predict(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_lr_predict, args)
}

pub fn shim_lr_train(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_lr_train, args)
}

pub fn shim_lr_evaluate(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_lr_evaluate, args)
}

pub fn shim_sgd(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sgd, args)
}

pub fn shim_sgd_step(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sgd_step, args)
}

pub fn shim_sgd_zero_grad(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sgd_zero_grad, args)
}

pub fn shim_adam(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_adam, args)
}

pub fn shim_adam_step(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_adam_step, args)
}

pub fn shim_mse_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_mse_loss, args)
}

pub fn shim_cross_entropy_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_cross_entropy_loss, args)
}

pub fn shim_binary_cross_entropy_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_binary_cross_entropy_loss, args)
}

pub fn shim_mae_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_mae_loss, args)
}

pub fn shim_huber_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_huber_loss, args)
}

pub fn shim_hinge_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_hinge_loss, args)
}

pub fn shim_kl_divergence(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_kl_divergence, args)
}

pub fn shim_smooth_l1_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_smooth_l1_loss, args)
}

pub fn shim_dataset(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_dataset, args)
}

pub fn shim_dataset_features(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_dataset_features, args)
}

pub fn shim_dataset_targets(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_dataset_targets, args)
}

pub fn shim_dataset_from_tensors(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_dataset_from_tensors, args)
}

pub fn shim_dataset_len(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_dataset_len, args)
}

pub fn shim_dataset_get(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_dataset_get, args)
}

pub fn shim_onehot(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_onehot, args)
}

pub fn shim_linear_layer(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_linear_layer, args)
}

pub fn shim_relu_layer(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_relu_layer, args)
}

pub fn shim_softmax_layer(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_softmax_layer, args)
}

pub fn shim_flatten_layer(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_flatten_layer, args)
}

pub fn shim_layer_call(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_layer_call, args)
}

pub fn shim_plugin_call(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_plugin_call, args)
}

pub fn shim_sequential(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sequential, args)
}

pub fn shim_sequential_add(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_sequential_add, args)
}

pub fn shim_neural_network(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_neural_network, args)
}

pub fn shim_nn_forward(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_forward, args)
}

/// Alias for scripts that call `ml.nn_forward` (trampolines must use a distinct fn path).
pub fn shim_nn_forward_alias(args: &[AbiValue]) -> AbiValue {
    shim_nn_forward(args)
}

pub fn shim_nn_train_sh(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_train_sh, args)
}

pub fn shim_nn_train(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_train, args)
}

pub fn shim_nn_save(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_save, args)
}

pub fn shim_nn_load(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_load, args)
}

pub fn shim_ml_save_model(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_ml_save_model, args)
}

pub fn shim_ml_load_model(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_ml_load_model, args)
}

pub fn shim_load_mnist(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_load_mnist, args)
}

pub fn shim_categorical_cross_entropy_loss(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_categorical_cross_entropy_loss, args)
}

pub fn shim_ml_set_device(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_ml_set_device, args)
}

pub fn shim_ml_get_device(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_ml_get_device, args)
}

pub fn shim_nn_set_device(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_set_device, args)
}

pub fn shim_nn_get_device(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_nn_get_device, args)
}

pub fn shim_devices(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_devices, args)
}

pub fn shim_available_backends(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_devices, args)
}

pub fn shim_ml_validate_model(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_ml_validate_model, args)
}

pub fn shim_ml_model_info(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_ml_model_info, args)
}

/// Alias for `ml_model_info` — `dc_fn!` needs a distinct function path for trampolines.
pub fn shim_model_info(args: &[AbiValue]) -> AbiValue {
    shim_ml_model_info(args)
}

pub fn shim_model_get_layer(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_model_get_layer, args)
}

pub fn shim_layer_freeze(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_layer_freeze, args)
}

pub fn shim_layer_unfreeze(args: &[AbiValue]) -> AbiValue {
    crate::abi_shim::shim_with(crate::natives::native_layer_unfreeze, args)
}

#[allow(unused_must_use)] // dc_fn! expands to a block whose `&mut m` borrow is intentionally unused
extern "C" fn register_ml_exports(ctx: *mut datacode_abi::VmContext) {
    let mut m = datacode_sdk::ModuleContext::new(ctx);
    datacode_sdk::dc_fn!(&mut m, "tensor", shim_tensor);
    datacode_sdk::dc_fn!(&mut m, "shape", shim_shape);
    datacode_sdk::dc_fn!(&mut m, "data", shim_data);
    datacode_sdk::dc_fn!(&mut m, "add", shim_add);
    datacode_sdk::dc_fn!(&mut m, "sub", shim_sub);
    datacode_sdk::dc_fn!(&mut m, "mul", shim_mul);
    datacode_sdk::dc_fn!(&mut m, "matmul", shim_matmul);
    datacode_sdk::dc_fn!(&mut m, "transpose", shim_transpose);
    datacode_sdk::dc_fn!(&mut m, "sum", shim_sum);
    datacode_sdk::dc_fn!(&mut m, "mean", shim_mean);
    datacode_sdk::dc_fn!(&mut m, "max_idx", shim_max_idx);
    datacode_sdk::dc_fn!(&mut m, "min_idx", shim_min_idx);
    datacode_sdk::dc_fn!(&mut m, "graph", shim_graph);
    datacode_sdk::dc_fn!(&mut m, "graph_add_input", shim_graph_add_input);
    datacode_sdk::dc_fn!(&mut m, "graph_add_op", shim_graph_add_op);
    datacode_sdk::dc_fn!(&mut m, "graph_forward", shim_graph_forward);
    datacode_sdk::dc_fn!(&mut m, "graph_get_output", shim_graph_get_output);
    datacode_sdk::dc_fn!(&mut m, "graph_backward", shim_graph_backward);
    datacode_sdk::dc_fn!(&mut m, "graph_get_gradient", shim_graph_get_gradient);
    datacode_sdk::dc_fn!(&mut m, "graph_zero_grad", shim_graph_zero_grad);
    datacode_sdk::dc_fn!(&mut m, "graph_set_requires_grad", shim_graph_set_requires_grad);
    datacode_sdk::dc_fn!(&mut m, "linear_regression", shim_linear_regression);
    datacode_sdk::dc_fn!(&mut m, "lr_predict", shim_lr_predict);
    datacode_sdk::dc_fn!(&mut m, "lr_train", shim_lr_train);
    datacode_sdk::dc_fn!(&mut m, "lr_evaluate", shim_lr_evaluate);
    datacode_sdk::dc_fn!(&mut m, "sgd", shim_sgd);
    datacode_sdk::dc_fn!(&mut m, "sgd_step", shim_sgd_step);
    datacode_sdk::dc_fn!(&mut m, "sgd_zero_grad", shim_sgd_zero_grad);
    datacode_sdk::dc_fn!(&mut m, "adam", shim_adam);
    datacode_sdk::dc_fn!(&mut m, "adam_step", shim_adam_step);
    datacode_sdk::dc_fn!(&mut m, "mse_loss", shim_mse_loss);
    datacode_sdk::dc_fn!(&mut m, "cross_entropy_loss", shim_cross_entropy_loss);
    datacode_sdk::dc_fn!(&mut m, "binary_cross_entropy_loss", shim_binary_cross_entropy_loss);
    datacode_sdk::dc_fn!(&mut m, "mae_loss", shim_mae_loss);
    datacode_sdk::dc_fn!(&mut m, "huber_loss", shim_huber_loss);
    datacode_sdk::dc_fn!(&mut m, "hinge_loss", shim_hinge_loss);
    datacode_sdk::dc_fn!(&mut m, "kl_divergence", shim_kl_divergence);
    datacode_sdk::dc_fn!(&mut m, "smooth_l1_loss", shim_smooth_l1_loss);
    datacode_sdk::dc_fn!(&mut m, "dataset", shim_dataset);
    datacode_sdk::dc_fn!(&mut m, "dataset_from_tensors", shim_dataset_from_tensors);
    datacode_sdk::dc_fn!(&mut m, "dataset_len", shim_dataset_len);
    datacode_sdk::dc_fn!(&mut m, "dataset_get", shim_dataset_get);
    datacode_sdk::dc_fn!(&mut m, "dataset_features", shim_dataset_features);
    datacode_sdk::dc_fn!(&mut m, "dataset_targets", shim_dataset_targets);
    datacode_sdk::dc_fn!(&mut m, "onehot", shim_onehot);
    datacode_sdk::dc_fn!(&mut m, "linear_layer", shim_linear_layer);
    datacode_sdk::dc_fn!(&mut m, "relu_layer", shim_relu_layer);
    datacode_sdk::dc_fn!(&mut m, "softmax_layer", shim_softmax_layer);
    datacode_sdk::dc_fn!(&mut m, "flatten_layer", shim_flatten_layer);
    datacode_sdk::dc_fn!(&mut m, "native_plugin_call", shim_plugin_call);
    datacode_sdk::dc_fn!(&mut m, "native_layer_call", shim_layer_call);
    datacode_sdk::dc_fn!(&mut m, "sequential", shim_sequential);
    datacode_sdk::dc_fn!(&mut m, "sequential_add", shim_sequential_add);
    datacode_sdk::dc_fn!(&mut m, "neural_network", shim_neural_network);
    datacode_sdk::dc_fn!(&mut m, "native_nn_forward", shim_nn_forward);
    datacode_sdk::dc_fn!(&mut m, "nn_forward", shim_nn_forward_alias);
    datacode_sdk::dc_fn!(&mut m, "nn_train_sh", shim_nn_train_sh);
    datacode_sdk::dc_fn!(&mut m, "nn_train", shim_nn_train);
    datacode_sdk::dc_fn!(&mut m, "nn_save", shim_nn_save);
    datacode_sdk::dc_fn!(&mut m, "nn_load", shim_nn_load);
    datacode_sdk::dc_fn!(&mut m, "ml_save_model", shim_ml_save_model);
    datacode_sdk::dc_fn!(&mut m, "ml_load_model", shim_ml_load_model);
    datacode_sdk::dc_fn!(&mut m, "load_mnist", shim_load_mnist);
    datacode_sdk::dc_fn!(&mut m, "categorical_cross_entropy_loss", shim_categorical_cross_entropy_loss);
    datacode_sdk::dc_fn!(&mut m, "ml_set_device", shim_ml_set_device);
    datacode_sdk::dc_fn!(&mut m, "ml_get_device", shim_ml_get_device);
    datacode_sdk::dc_fn!(&mut m, "nn_set_device", shim_nn_set_device);
    datacode_sdk::dc_fn!(&mut m, "nn_get_device", shim_nn_get_device);
    datacode_sdk::dc_fn!(&mut m, "devices", shim_devices);
    datacode_sdk::dc_fn!(&mut m, "available_backends", shim_available_backends);
    datacode_sdk::dc_fn!(&mut m, "ml_validate_model", shim_ml_validate_model);
    datacode_sdk::dc_fn!(&mut m, "ml_model_info", shim_ml_model_info);
    datacode_sdk::dc_fn!(&mut m, "model_info", shim_model_info);
    datacode_sdk::dc_fn!(&mut m, "model_get_layer", shim_model_get_layer);
    datacode_sdk::dc_fn!(&mut m, "layer_freeze", shim_layer_freeze);
    datacode_sdk::dc_fn!(&mut m, "layer_unfreeze", shim_layer_unfreeze);
}

datacode_sdk::define_module!("ml", 1, 3, register_ml_exports);