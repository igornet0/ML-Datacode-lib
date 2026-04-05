// ML library for DataCode (extracted from the VM crate).
// Module implementations live under `src/{vm,core,engine,nn,gpu}/`; paths here preserve `ml::...` API.

#[path = "vm/ml_types.rs"]
pub mod ml_types;
#[path = "vm/plugin_opaque.rs"]
pub mod plugin_opaque;
#[path = "vm/native_error.rs"]
pub mod native_error;
#[path = "vm/runtime.rs"]
pub mod runtime;
#[path = "vm/natives.rs"]
pub mod natives;
#[path = "vm/vm_value.rs"]
mod vm_value;
#[path = "vm/plugin_abi_bridge.rs"]
mod plugin_abi_bridge;
#[path = "vm/abi_shim.rs"]
mod abi_shim;
#[path = "vm/module_entry.rs"]
mod module_entry;

#[path = "core/tensor.rs"]
pub mod tensor;
#[path = "nn/dataset.rs"]
pub mod dataset;
#[path = "engine/autograd.rs"]
pub mod autograd;
#[path = "engine/ops.rs"]
pub mod ops;
#[path = "engine/graph.rs"]
pub mod graph;
#[path = "nn/model.rs"]
pub mod model;
#[path = "nn/optimizer.rs"]
pub mod optimizer;
#[path = "nn/loss.rs"]
pub mod loss;
#[path = "nn/layer.rs"]
pub mod layer;
#[path = "core/device.rs"]
pub mod device;
#[path = "nn/scheduler.rs"]
pub mod scheduler;
#[path = "core/tensor_pool.rs"]
pub mod tensor_pool;
#[path = "core/gpu_cache.rs"]
pub mod gpu_cache;
#[path = "core/context.rs"]
pub mod context;
#[cfg(feature = "gpu")]
#[path = "gpu/ops_gpu.rs"]
pub mod ops_gpu;

#[cfg(feature = "gpu")]
#[path = "gpu/candle_integration.rs"]
mod candle_integration;

#[path = "core/backend_registry.rs"]
pub mod backend_registry;

#[path = "core/mnist_paths.rs"]
pub mod mnist_paths;
#[path = "core/mnist_locate.rs"]
mod mnist_locate;

pub use autograd::{Variable, requires_grad};
pub use graph::{Graph, Node, NodeId, OpType};
pub use model::{LinearRegression, NeuralNetwork};
pub use optimizer::{SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdamW, OptimizerType};
pub use loss::{
    binary_cross_entropy_loss, categorical_cross_entropy_loss, hinge_loss, huber_loss,
    kl_divergence, mae_loss, mse_loss, smooth_l1_loss, sparse_softmax_cross_entropy_loss,
};
pub use layer::{
    add_layer_to_registry, forward_layer_var, Layer, LayerId, Linear, ReLU, Sequential, Sigmoid,
    Softmax, Tanh, Flatten, with_layer,
};
pub use device::Device;
pub use gpu_cache::{
    clear_global_gpu_cache, get_gpu_cache_stats, get_gpu_tensor_from_cache, init_global_gpu_cache,
    update_gpu_tensor_in_cache, GpuTensorCache,
};
pub use tensor_pool::{
    clear_global_pool, get_pool_stats, get_tensor_from_pool, return_tensor_to_pool, TensorPool,
};
pub use context::MlContext;
pub use tensor::Tensor;
pub use backend_registry::BackendRegistry;

pub use datacode_abi;
pub use datacode_sdk;
pub use ml_types::{MlHandle, MlValueKind};
pub use vm_value::Value as PluginValue;
