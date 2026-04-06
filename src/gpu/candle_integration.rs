#![cfg(feature = "gpu")]

// Candle integration for training neural networks
// This module provides conversion between DataCode types and Candle types

use crate::tensor::Tensor as DataCodeTensor;
use crate::layer::Sequential as DataCodeSequential;
use crate::layer::LayerType as DataCodeLayerType;

fn candle_dim_for_axis(axis: usize, rank: usize) -> usize {
    if rank == 0 {
        return 0;
    }
    axis.min(rank.saturating_sub(1))
}

fn slice_var_set_from_dc(
    var: &candle_core::Var,
    dc: &DataCodeTensor,
    device: &candle_core::Device,
) -> Result<(), String> {
    use candle_core::D;
    let new_t = to_candle_tensor(dc, device)?;
    let target: &candle_core::Tensor = var;
    let new_t = new_t
        .to_device(target.device())
        .map_err(|e| format!("slice_var_set_from_dc: device: {}", e))?;
    if new_t.dims() != target.dims() {
        return Err(format!(
            "slice_var_set_from_dc: shape mismatch {:?} vs {:?}",
            new_t.dims(),
            target.dims()
        ));
    }
    var.slice_set(&new_t, D::Minus1, 0)
        .map_err(|e| format!("slice_var_set_from_dc: {}", e))
}

/// Convert DataCode Tensor to Candle Tensor
pub fn to_candle_tensor(
    dc_tensor: &DataCodeTensor,
    device: &candle_core::Device,
) -> Result<candle_core::Tensor, String> {
    use candle_core::Shape;
    
    // Ensure tensor is on CPU for data access
    let dc_tensor_cpu = dc_tensor.to_cpu()
        .map_err(|e| format!("Failed to move tensor to CPU: {}", e))?;
    
    // Get shape and data
    let shape = dc_tensor_cpu.shape();
    let data_array = dc_tensor_cpu.data();
    
    // Convert ArrayD<f32> to &[f32] using as_slice()
    let data_slice: &[f32] = data_array.as_slice()
        .ok_or_else(|| "Failed to convert tensor data to slice. Tensor may not be contiguous.".to_string())?;
    
    // Create Candle shape from dimensions
    let candle_shape = Shape::from_dims(shape);
    
    // Create Candle tensor on target device
    let tensor = candle_core::Tensor::from_slice(data_slice, candle_shape, device)
        .map_err(|e| format!("Failed to create candle tensor: {}", e))?;
    
    Ok(tensor)
}

/// Convert Candle Tensor to DataCode Tensor
pub fn from_candle_tensor(ct: &candle_core::Tensor) -> Result<DataCodeTensor, String> {
    use candle_core::Device;
    
    // Move tensor to CPU if needed for data access
    let ct_cpu = ct.to_device(&Device::Cpu)
        .map_err(|e| format!("Failed to move candle tensor to CPU: {}", e))?;
    
    // Get shape and convert to Vec<usize>
    let shape = ct_cpu.dims().to_vec();
    let rank = shape.len();
    
    // Convert tensor to Vec<f32> based on rank
    let data = if rank == 1 {
        // 1D tensor, use to_vec1 directly
        ct_cpu.to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?
    } else if rank == 2 {
        // 2D tensor, use to_vec2 and flatten
        let data_2d = ct_cpu.to_vec2::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?;
        data_2d.into_iter().flat_map(|row| row.into_iter()).collect()
    } else {
        // Higher rank tensor, reshape to 1D first
        let total_size: usize = shape.iter().product();
        let flattened = ct_cpu.reshape((total_size,))
            .map_err(|e| format!("Failed to reshape tensor to 1D: {}", e))?;
        flattened.to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert candle tensor to Vec: {}", e))?
    };
    
    // Create DataCode tensor
    let result = DataCodeTensor::new(data, shape.clone())
        .map_err(|e| format!("Failed to create DataCode tensor with shape {:?}: {}", shape, e))?;
    
    Ok(result)
}

/// Convert DataCode Sequential to Candle Sequential
/// This creates a NEW Candle Sequential with copied weights
/// Returns (Sequential, VarMap) - weights are set after layer creation
pub fn to_candle_sequential(
    dc_seq: &DataCodeSequential,
    device: &candle_core::Device,
) -> Result<(candle_nn::Sequential, candle_nn::VarMap), String> {
    use candle_core::DType;
    use candle_nn::activation::Activation;
    use candle_nn::{linear_b, VarBuilder, VarMap};
    
    // Create VarMap to store parameters
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let mut candle_seq = candle_nn::seq();
    // Index for `vb.pp(format!("layer{}", param_layer_idx))` for every layer that registers Vars.
    let mut param_layer_idx: usize = 0;
    
    // Check if we have layers or need to use layer_ids
    let has_layers = !dc_seq.layers.is_empty();
    
    if !has_layers && dc_seq.layer_ids.is_empty() {
        return Err("Sequential has no layers to convert".to_string());
    }
    
    // Process layers
    if has_layers {
        for layer_type in &dc_seq.layers {
            match layer_type {
                DataCodeLayerType::Linear(linear) => {
                    let in_features = linear.in_features;
                    let out_features = linear.out_features;
                    let use_bias = linear.bias.is_some();
                    let layer_vb = vb.pp(format!("layer{}", param_layer_idx));
                    param_layer_idx += 1;
                    let linear_candle = linear_b(in_features, out_features, use_bias, layer_vb)
                        .map_err(|e| format!("Failed to create Candle linear layer: {}", e))?;
                    candle_seq = candle_seq.add(linear_candle);
                }
                DataCodeLayerType::ReLU(_) => {
                    candle_seq = candle_seq.add(Activation::Relu);
                }
                DataCodeLayerType::Sigmoid(_) => {
                    candle_seq = candle_seq.add(Activation::Sigmoid);
                }
                DataCodeLayerType::Tanh(_) => {
                    candle_seq = candle_seq.add_fn(|x| Ok(x.tanh()?));
                }
                DataCodeLayerType::Softmax(s) => {
                    use candle_nn::ops::softmax;
                    let axis = s.axis;
                    candle_seq = candle_seq.add_fn(move |x| {
                        let rank = x.dims().len();
                        softmax(x, candle_dim_for_axis(axis, rank))
                    });
                }
                DataCodeLayerType::Flatten(_) => {}
                DataCodeLayerType::LogSoftmax(l) => {
                    use candle_nn::ops::log_softmax;
                    let axis = l.axis;
                    candle_seq = candle_seq.add_fn(move |x| {
                        let rank = x.dims().len();
                        log_softmax(x, candle_dim_for_axis(axis, rank))
                    });
                }
                DataCodeLayerType::Gelu(_) => {
                    candle_seq = candle_seq.add(Activation::Gelu);
                }
                DataCodeLayerType::Softplus(_) => {
                    candle_seq = candle_seq.add_fn(|x| {
                        let one = candle_core::Tensor::ones_like(x)?;
                        x.exp()?.broadcast_add(&one)?.log()
                    });
                }
                DataCodeLayerType::Elu(e) => {
                    let alpha = e.alpha as f64;
                    candle_seq = candle_seq.add(Activation::Elu(alpha));
                }
                DataCodeLayerType::Selu(_) => {
                    use candle_nn::ops::selu;
                    let alpha = crate::ops::SELU_ALPHA;
                    let gamma = crate::ops::SELU_SCALE;
                    candle_seq = candle_seq.add_fn(move |x| selu(x, alpha, gamma));
                }
                DataCodeLayerType::PReLU(_) => {
                    let layer_vb = vb.pp(format!("layer{}", param_layer_idx));
                    param_layer_idx += 1;
                    let prelu_candle = candle_nn::prelu(None, layer_vb)
                        .map_err(|e| format!("Failed to create Candle PReLU: {}", e))?;
                    candle_seq = candle_seq.add(prelu_candle);
                }
                DataCodeLayerType::Dropout(d) => {
                    use candle_nn::ops::dropout;
                    let p = d.p;
                    candle_seq = candle_seq.add_fn(move |x| dropout(x, p));
                }
                DataCodeLayerType::Dropout2d(d) => {
                    let p = d.p;
                    candle_seq = candle_seq.add_fn(move |x| {
                        use candle_core::Tensor;
                        use candle_nn::ops::dropout;
                        let dims = x.dims();
                        if dims.len() != 4 {
                            return dropout(x, p);
                        }
                        let (n, c, _h, _w) = (dims[0], dims[1], dims[2], dims[3]);
                        if !(0.0..1.0).contains(&p) {
                            return Err(candle_core::Error::Msg(format!(
                                "dropout2d: p must be in [0,1), got {}",
                                p
                            )));
                        }
                        let scale = 1.0 / (1.0 - p as f64);
                        let rand = Tensor::rand(0f32, 1f32, (n, c, 1, 1), x.device())?;
                        let drop_p = Tensor::full(p, (n, c, 1, 1), x.device())?;
                        let mask = (rand.ge(&drop_p)?.to_dtype(x.dtype())? * scale)?;
                        x.broadcast_mul(&mask)
                    });
                }
                DataCodeLayerType::DropConnect(d) => {
                    use candle_nn::ops::dropout;
                    let p = d.p;
                    candle_seq = candle_seq.add_fn(move |x| dropout(x, p));
                }
                DataCodeLayerType::Conv2d(conv) => {
                    if conv.stride.0 != conv.stride.1 || conv.padding.0 != conv.padding.1 {
                        return Err(
                            "Candle Conv2d: нужны симметричные stride и padding (sx==sy, px==py). Используйте CPU или задайте одинаковые значения по осям."
                                .to_string(),
                        );
                    }
                    let w = conv.weight.data.borrow().clone().to_cpu().map_err(|e| e.to_string())?;
                    let shape = w.shape();
                    if shape.len() != 4 {
                        return Err(format!("Conv2d: ожидался вес ранга 4, получили {:?}", shape));
                    }
                    let (out_c, in_c, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
                    if kh != kw {
                        return Err(
                            "Candle conv2d: квадратное ядро (kh==kw). Используйте CPU или квадратный kernel."
                                .to_string(),
                        );
                    }
                    let cfg = candle_nn::Conv2dConfig {
                        padding: conv.padding.0,
                        stride: conv.stride.0,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    };
                    let layer_vb = vb.pp(format!("layer{}", param_layer_idx));
                    param_layer_idx += 1;
                    let conv_candle = if conv.bias.is_some() {
                        candle_nn::conv2d(in_c, out_c, kh, cfg, layer_vb)
                    } else {
                        candle_nn::conv2d_no_bias(in_c, out_c, kh, cfg, layer_vb)
                    }
                    .map_err(|e| format!("Failed to create Candle Conv2d: {}", e))?;
                    candle_seq = candle_seq.add(conv_candle);
                }
                DataCodeLayerType::MaxPool2d(pool) => {
                    let kh = pool.kh;
                    let kw = pool.kw;
                    let sy = pool.sy;
                    let sx = pool.sx;
                    candle_seq = candle_seq.add_fn(move |x| {
                        x.max_pool2d_with_stride((kh, kw), (sy, sx))
                    });
                }
                DataCodeLayerType::Conv1d(_)
                | DataCodeLayerType::MaxPool1d(_)
                | DataCodeLayerType::AvgPool1d(_)
                | DataCodeLayerType::AvgPool2d(_)
                | DataCodeLayerType::GlobalMaxPool2d(_)
                | DataCodeLayerType::GlobalAvgPool2d(_) => {
                    return Err(
                        "Candle/Metal: этот тип слоя пока не поддерживается в to_candle_sequential. Используйте CPU train_sh."
                            .to_string(),
                    );
                }
                DataCodeLayerType::Placeholder(_) => {}
            }
        }
    } else {
        use crate::layer::with_layer;
        for &layer_id in &dc_seq.layer_ids {
            let head = with_layer(layer_id, |layer| {
                let params = layer.parameters_var();
                let dbg = format!("{:?}", layer);
                (
                    !params.is_empty(),
                    params,
                    dbg,
                    layer.in_features(),
                    layer.out_features(),
                )
            })
            .ok_or_else(|| format!("Missing layer_id {}", layer_id))?;

            let (has_params, params, dbg, in_f, out_f) = head;

            if has_params {
                if dbg.contains("Conv2d") {
                    return Err(
                        "Candle: Conv2d через только layer_ids не поддерживается — соберите Sequential с полем layers."
                            .to_string(),
                    );
                }
                if dbg.contains("PReLU") && params.len() == 1 {
                    let wshape = params[0].data.borrow().shape().to_vec();
                    if wshape == [1] {
                        let layer_vb = vb.pp(format!("layer{}", param_layer_idx));
                        param_layer_idx += 1;
                        let prelu_candle = candle_nn::prelu(None, layer_vb)
                            .map_err(|e| format!("Failed to create Candle PReLU: {}", e))?;
                        candle_seq = candle_seq.add(prelu_candle);
                        continue;
                    }
                }
                if in_f > 0 && out_f > 0 {
                    let use_bias = params.len() >= 2;
                    let layer_vb = vb.pp(format!("layer{}", param_layer_idx));
                    param_layer_idx += 1;
                    let linear_candle = linear_b(in_f, out_f, use_bias, layer_vb)
                        .map_err(|e| format!("Failed to create Candle linear layer: {}", e))?;
                    candle_seq = candle_seq.add(linear_candle);
                    continue;
                }
                return Err(format!(
                    "Candle: слой с параметрами (layer_id={}) не поддерживается через layer_ids без поля layers.",
                    layer_id
                ));
            }

            if dbg.contains("LogSoftmax") {
                use candle_nn::ops::log_softmax;
                candle_seq = candle_seq.add_fn(|x| log_softmax(x, candle_core::D::Minus1));
            } else if dbg.contains("Softmax") {
                use candle_nn::ops::softmax;
                candle_seq = candle_seq.add_fn(|x| softmax(x, candle_core::D::Minus1));
            } else if dbg.contains("Gelu") {
                candle_seq = candle_seq.add(Activation::Gelu);
            } else if dbg.contains("Softplus") {
                candle_seq = candle_seq.add_fn(|x| {
                    let one = candle_core::Tensor::ones_like(x)?;
                    x.exp()?.broadcast_add(&one)?.log()
                });
            } else if dbg.contains("Elu") {
                candle_seq = candle_seq.add(Activation::Elu(1.0));
            } else if dbg.contains("Selu") {
                use candle_nn::ops::selu;
                let alpha = crate::ops::SELU_ALPHA;
                let gamma = crate::ops::SELU_SCALE;
                candle_seq = candle_seq.add_fn(move |x| selu(x, alpha, gamma));
            } else if dbg.contains("ReLU") {
                candle_seq = candle_seq.add(Activation::Relu);
            } else if dbg.contains("Sigmoid") {
                candle_seq = candle_seq.add(Activation::Sigmoid);
            } else if dbg.contains("Tanh") {
                candle_seq = candle_seq.add_fn(|x| Ok(x.tanh()?));
            } else if dbg.contains("Flatten") {
            } else if dbg.contains("Dropout2d") {
                let p = parse_debug_f32_after_key(&dbg, "p:").unwrap_or(0.5);
                candle_seq = candle_seq.add_fn(move |x| {
                    use candle_core::Tensor;
                    use candle_nn::ops::dropout;
                    let dims = x.dims();
                    if dims.len() != 4 {
                        return dropout(x, p);
                    }
                    let (n, c, _h, _w) = (dims[0], dims[1], dims[2], dims[3]);
                    if !(0.0..1.0).contains(&p) {
                        return Err(candle_core::Error::msg(format!(
                            "dropout2d: p must be in [0,1), got {}",
                            p
                        )));
                    }
                    let scale = 1.0 / (1.0 - p as f64);
                    let rand = Tensor::rand(0f32, 1f32, (n, c, 1, 1), x.device())?;
                    let drop_p = Tensor::full(p, (n, c, 1, 1), x.device())?;
                    let mask = (rand.ge(&drop_p)?.to_dtype(x.dtype())? * scale)?;
                    x.broadcast_mul(&mask)
                });
            } else if dbg.contains("DropConnect") || dbg.contains("Dropout") {
                let p = parse_debug_f32_after_key(&dbg, "p:").unwrap_or(0.5);
                use candle_nn::ops::dropout;
                candle_seq = candle_seq.add_fn(move |x| dropout(x, p));
            } else if dbg.contains("MaxPool2d") {
                return Err(
                    "Candle: MaxPool2d через только layer_ids не поддерживается — используйте поле layers."
                        .to_string(),
                );
            } else {
                return Err(format!("Unknown layer type for layer_id {}: {}", layer_id, dbg));
            }
        }
    }

    Ok((candle_seq, varmap))
}

fn parse_debug_f32_after_key(s: &str, key: &str) -> Option<f32> {
    let pos = s.find(key)?;
    let tail = &s[pos + key.len()..];
    let num: String = tail
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    num.parse().ok()
}

/// Candle `VarMap::all_vars()` returns `HashMap` values in arbitrary order — always look up by path.
fn varmap_get_var(varmap: &candle_nn::VarMap, path: &str) -> Result<candle_core::Var, String> {
    let map = varmap
        .data()
        .lock()
        .map_err(|e| format!("VarMap lock: {}", e))?;
    map.get(path)
        .cloned()
        .ok_or_else(|| format!("VarMap: missing variable '{}'", path))
}

fn varmap_len(varmap: &candle_nn::VarMap) -> Result<usize, String> {
    let map = varmap
        .data()
        .lock()
        .map_err(|e| format!("VarMap lock: {}", e))?;
    Ok(map.len())
}

fn copy_weights_from_candle_layers(
    varmap: &candle_nn::VarMap,
    dc_seq: &DataCodeSequential,
) -> Result<(), String> {
    use crate::layer::with_layer;
    let mut param_layer_idx: usize = 0;
    let mut n_read = 0usize;
    for (i, layer_type) in dc_seq.layers.iter().enumerate() {
        let layer_id = dc_seq.layer_ids.get(i).copied();
        match layer_type {
            DataCodeLayerType::Linear(linear) => {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let w_var = varmap_get_var(varmap, &format!("{prefix}.weight"))?;
                let weight_dc = from_candle_tensor(w_var.as_tensor())?;
                n_read += 1;
                let mut bias_dc_opt = None;
                if linear.bias.is_some() {
                    let b_var = varmap_get_var(varmap, &format!("{prefix}.bias"))?;
                    bias_dc_opt = Some(from_candle_tensor(b_var.as_tensor())?);
                    n_read += 1;
                }
                if let Some(lid) = layer_id {
                    with_layer(lid, |layer| {
                        let params = layer.parameters_var();
                        if !params.is_empty() {
                            *params[0].data.borrow_mut() = weight_dc.clone();
                        }
                        if let Some(ref b) = bias_dc_opt {
                            if params.len() >= 2 {
                                *params[1].data.borrow_mut() = b.clone();
                            }
                        }
                    });
                }
            }
            DataCodeLayerType::Conv2d(conv) => {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let w_var = varmap_get_var(varmap, &format!("{prefix}.weight"))?;
                let weight_dc = from_candle_tensor(w_var.as_tensor())?;
                n_read += 1;
                let mut bias_dc_opt = None;
                if conv.bias.is_some() {
                    let b_var = varmap_get_var(varmap, &format!("{prefix}.bias"))?;
                    bias_dc_opt = Some(from_candle_tensor(b_var.as_tensor())?);
                    n_read += 1;
                }
                if let Some(lid) = layer_id {
                    with_layer(lid, |layer| {
                        let params = layer.parameters_var();
                        if !params.is_empty() {
                            *params[0].data.borrow_mut() = weight_dc.clone();
                        }
                        if let Some(ref b) = bias_dc_opt {
                            if params.len() >= 2 {
                                *params[1].data.borrow_mut() = b.clone();
                            }
                        }
                    });
                }
            }
            DataCodeLayerType::Conv1d(_)
            | DataCodeLayerType::MaxPool1d(_)
            | DataCodeLayerType::AvgPool1d(_)
            | DataCodeLayerType::AvgPool2d(_)
            | DataCodeLayerType::GlobalMaxPool2d(_)
            | DataCodeLayerType::GlobalAvgPool2d(_) => {
                return Err(
                    "Candle/Metal: этот тип слоя пока не поддерживается в copy_weights_from_candle."
                        .to_string(),
                );
            }
            DataCodeLayerType::PReLU(_) => {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let a_var = varmap_get_var(varmap, &format!("{prefix}.weight"))?;
                let alpha_dc = from_candle_tensor(a_var.as_tensor())?;
                n_read += 1;
                if let Some(lid) = layer_id {
                    with_layer(lid, |layer| {
                        let params = layer.parameters_var();
                        if !params.is_empty() {
                            *params[0].data.borrow_mut() = alpha_dc.clone();
                        }
                    });
                }
            }
            _ => {}
        }
    }
    let total = varmap_len(varmap)?;
    if n_read != total {
        return Err(format!(
            "copy_weights_from_candle: ожидалось {} параметров в VarMap, прочитано {}",
            total, n_read
        ));
    }
    Ok(())
}

fn copy_weights_from_candle_layer_ids(
    varmap: &candle_nn::VarMap,
    dc_seq: &DataCodeSequential,
) -> Result<(), String> {
    use crate::layer::with_layer;
    let mut param_layer_idx: usize = 0;
    let mut n_read = 0usize;
    for &layer_id in &dc_seq.layer_ids {
        let head = with_layer(layer_id, |layer| {
            let params = layer.parameters_var();
            let dbg = format!("{:?}", layer);
            (
                !params.is_empty(),
                params,
                dbg,
                layer.in_features(),
                layer.out_features(),
            )
        })
        .ok_or_else(|| format!("copy_weights_from_candle: нет слоя {}", layer_id))?;
        let (has_params, params, dbg, in_f, out_f) = head;
        if !has_params {
            continue;
        }
        if dbg.contains("Conv2d") {
            return Err(
                "copy_weights_from_candle: Conv2d только через поле layers поддерживается.".to_string(),
            );
        }
        if dbg.contains("PReLU") && params.len() == 1 {
            let wshape = params[0].data.borrow().shape().to_vec();
            if wshape == [1] {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let a_var = varmap_get_var(varmap, &format!("{prefix}.weight"))?;
                let alpha_dc = from_candle_tensor(a_var.as_tensor())?;
                n_read += 1;
                with_layer(layer_id, |layer| {
                    let p = layer.parameters_var();
                    if !p.is_empty() {
                        *p[0].data.borrow_mut() = alpha_dc.clone();
                    }
                });
                continue;
            }
        }
        if in_f > 0 && out_f > 0 {
            let prefix = format!("layer{}", param_layer_idx);
            param_layer_idx += 1;
            let w_var = varmap_get_var(varmap, &format!("{prefix}.weight"))?;
            let weight_dc = from_candle_tensor(w_var.as_tensor())?;
            n_read += 1;
            let mut bias_dc_opt = None;
            if params.len() >= 2 {
                let b_var = varmap_get_var(varmap, &format!("{prefix}.bias"))?;
                bias_dc_opt = Some(from_candle_tensor(b_var.as_tensor())?);
                n_read += 1;
            }
            with_layer(layer_id, |layer| {
                let p = layer.parameters_var();
                if !p.is_empty() {
                    *p[0].data.borrow_mut() = weight_dc.clone();
                }
                if let Some(ref b) = bias_dc_opt {
                    if p.len() >= 2 {
                        *p[1].data.borrow_mut() = b.clone();
                    }
                }
            });
            continue;
        }
        return Err(format!(
            "copy_weights_from_candle: неизвестный слой с параметрами (layer_id={}): {}",
            layer_id, dbg
        ));
    }
    let total = varmap_len(varmap)?;
    if n_read != total {
        return Err(format!(
            "copy_weights_from_candle (layer_ids): ожидалось {} параметров в VarMap, прочитано {}",
            total, n_read
        ));
    }
    Ok(())
}

/// Copy weights from Candle VarMap back to DataCode Sequential
pub fn copy_weights_from_candle(
    varmap: &candle_nn::VarMap,
    dc_seq: &mut DataCodeSequential,
) -> Result<(), String> {
    if !dc_seq.layers.is_empty() {
        copy_weights_from_candle_layers(varmap, dc_seq)
    } else if !dc_seq.layer_ids.is_empty() {
        copy_weights_from_candle_layer_ids(varmap, dc_seq)
    } else {
        Ok(())
    }
}

fn copy_weights_to_candle_layers(
    varmap: &candle_nn::VarMap,
    dc_seq: &DataCodeSequential,
    device: &candle_core::Device,
) -> Result<(), String> {
    let mut param_layer_idx: usize = 0;
    let mut n_written = 0usize;
    for layer_type in &dc_seq.layers {
        match layer_type {
            DataCodeLayerType::Linear(linear) => {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let w_dc = linear
                    .weight
                    .data
                    .borrow()
                    .clone()
                    .to_cpu()
                    .map_err(|e| e)?;
                slice_var_set_from_dc(
                    &varmap_get_var(varmap, &format!("{prefix}.weight"))?,
                    &w_dc,
                    device,
                )?;
                n_written += 1;
                if linear.bias.is_some() {
                    let b_dc = linear
                        .bias
                        .as_ref()
                        .unwrap()
                        .data
                        .borrow()
                        .clone()
                        .to_cpu()
                        .map_err(|e| e)?;
                    slice_var_set_from_dc(
                        &varmap_get_var(varmap, &format!("{prefix}.bias"))?,
                        &b_dc,
                        device,
                    )?;
                    n_written += 1;
                }
            }
            DataCodeLayerType::Conv2d(conv) => {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let w_dc = conv
                    .weight
                    .data
                    .borrow()
                    .clone()
                    .to_cpu()
                    .map_err(|e| e)?;
                slice_var_set_from_dc(
                    &varmap_get_var(varmap, &format!("{prefix}.weight"))?,
                    &w_dc,
                    device,
                )?;
                n_written += 1;
                if conv.bias.is_some() {
                    let b_dc = conv
                        .bias
                        .as_ref()
                        .unwrap()
                        .data
                        .borrow()
                        .clone()
                        .to_cpu()
                        .map_err(|e| e)?;
                    slice_var_set_from_dc(
                        &varmap_get_var(varmap, &format!("{prefix}.bias"))?,
                        &b_dc,
                        device,
                    )?;
                    n_written += 1;
                }
            }
            DataCodeLayerType::Conv1d(_)
            | DataCodeLayerType::MaxPool1d(_)
            | DataCodeLayerType::AvgPool1d(_)
            | DataCodeLayerType::AvgPool2d(_)
            | DataCodeLayerType::GlobalMaxPool2d(_)
            | DataCodeLayerType::GlobalAvgPool2d(_) => {
                return Err(
                    "Candle/Metal: этот тип слоя пока не поддерживается в copy_weights_to_candle."
                        .to_string(),
                );
            }
            DataCodeLayerType::PReLU(p) => {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let a_dc = p
                    .alpha
                    .data
                    .borrow()
                    .clone()
                    .to_cpu()
                    .map_err(|e| e)?;
                slice_var_set_from_dc(
                    &varmap_get_var(varmap, &format!("{prefix}.weight"))?,
                    &a_dc,
                    device,
                )?;
                n_written += 1;
            }
            _ => {}
        }
    }
    let total = varmap_len(varmap)?;
    if n_written != total {
        return Err(format!(
            "copy_weights_to_candle: ожидалось {} параметров в VarMap, записано {}",
            total, n_written
        ));
    }
    Ok(())
}

fn copy_weights_to_candle_layer_ids(
    varmap: &candle_nn::VarMap,
    dc_seq: &DataCodeSequential,
    device: &candle_core::Device,
) -> Result<(), String> {
    use crate::layer::with_layer;
    let mut param_layer_idx: usize = 0;
    let mut n_written = 0usize;
    for &layer_id in &dc_seq.layer_ids {
        let head = with_layer(layer_id, |layer| {
            let params = layer.parameters_var();
            let dbg = format!("{:?}", layer);
            (
                !params.is_empty(),
                params,
                dbg,
                layer.in_features(),
                layer.out_features(),
            )
        })
        .ok_or_else(|| format!("copy_weights_to_candle: нет слоя {}", layer_id))?;
        let (has_params, params, dbg, in_f, out_f) = head;
        if !has_params {
            continue;
        }
        if dbg.contains("Conv2d") {
            return Err(
                "copy_weights_to_candle: Conv2d только через поле layers поддерживается.".to_string(),
            );
        }
        if dbg.contains("PReLU") && params.len() == 1 {
            let wshape = params[0].data.borrow().shape().to_vec();
            if wshape == [1] {
                let prefix = format!("layer{}", param_layer_idx);
                param_layer_idx += 1;
                let a_dc = params[0].data.borrow().clone().to_cpu().map_err(|e| e)?;
                slice_var_set_from_dc(
                    &varmap_get_var(varmap, &format!("{prefix}.weight"))?,
                    &a_dc,
                    device,
                )?;
                n_written += 1;
                continue;
            }
        }
        if in_f > 0 && out_f > 0 {
            let prefix = format!("layer{}", param_layer_idx);
            param_layer_idx += 1;
            let w_dc = params[0].data.borrow().clone().to_cpu().map_err(|e| e)?;
            slice_var_set_from_dc(
                &varmap_get_var(varmap, &format!("{prefix}.weight"))?,
                &w_dc,
                device,
            )?;
            n_written += 1;
            if params.len() >= 2 {
                let b_dc = params[1].data.borrow().clone().to_cpu().map_err(|e| e)?;
                slice_var_set_from_dc(
                    &varmap_get_var(varmap, &format!("{prefix}.bias"))?,
                    &b_dc,
                    device,
                )?;
                n_written += 1;
            }
            continue;
        }
        return Err(format!(
            "copy_weights_to_candle: неизвестный слой с параметрами (layer_id={}): {}",
            layer_id, dbg
        ));
    }
    let total = varmap_len(varmap)?;
    if n_written != total {
        return Err(format!(
            "copy_weights_to_candle (layer_ids): ожидалось {} параметров в VarMap, записано {}",
            total, n_written
        ));
    }
    Ok(())
}

/// Copy weights from DataCode Sequential to Candle VarMap
/// This should be called after to_candle_sequential to initialize weights
/// before training starts
pub fn copy_weights_to_candle(
    varmap: &candle_nn::VarMap,
    dc_seq: &DataCodeSequential,
    device: &candle_core::Device,
) -> Result<(), String> {
    if !dc_seq.layers.is_empty() {
        copy_weights_to_candle_layers(varmap, dc_seq, device)
    } else if !dc_seq.layer_ids.is_empty() {
        copy_weights_to_candle_layer_ids(varmap, dc_seq, device)
    } else {
        Ok(())
    }
}

