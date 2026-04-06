// Basic operations for ML module
// Rewritten to match MetalNN architecture

use crate::tensor::Tensor;

/// Сложение двух тензоров
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    
    // Проверяем, нужен ли broadcast
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Если формы одинаковые, просто складываем
    if a_shape == b_shape {
        let result = a_arr + b_arr;
        return Tensor::from_array(result);
    }
    
    // Если b - это 1D тензор, который нужно broadcast к a
    // Например: a = [batch, features], b = [features] -> broadcast b к [1, features]
    if b_shape.len() == 1 && a_shape.len() == 2 && b_shape[0] == a_shape[1] {
        // Broadcast b: [features] -> [1, features] -> [batch, features]
        let b_2d = b_arr.clone().insert_axis(ndarray::Axis(0));
        let b_broadcast = b_2d.broadcast(a_shape).expect("Failed to broadcast").to_owned();
        let result = a_arr + &b_broadcast;
        return Tensor::from_array(result);
    }
    
    // Пробуем автоматический broadcast через ndarray
    let result = a_arr + b_arr;
    Tensor::from_array(result)
}

/// Умножение двух тензоров (элементное)
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    let result = a_arr * b_arr;
    Tensor::from_array(result)
}

/// Скалярное умножение
pub fn scalar_mul(a: &Tensor, scalar: f32) -> Tensor {
    let a_arr = a.data();
    let result = a_arr * scalar;
    Tensor::from_array(result)
}

/// Деление двух тензоров (элементное)
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    let result = a_arr / b_arr;
    Tensor::from_array(result)
}

/// Деление тензора на скаляр
pub fn scalar_div(a: &Tensor, scalar: f32) -> Tensor {
    let a_arr = a.data();
    let result = a_arr / scalar;
    Tensor::from_array(result)
}

/// Квадратный корень (элементный)
pub fn sqrt(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| x.sqrt());
    Tensor::from_array(result)
}

/// Матричное умножение (оптимизированная версия)
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_arr = a.data();
    let b_arr = b.data();
    
    // Оптимизированная реализация для 2D тензоров
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let (_m, k) = (a_shape[0], a_shape[1]);
        let (k2, _n) = (b_shape[0], b_shape[1]);
        
        assert_eq!(k, k2, "Matrix dimensions don't match for multiplication");
        
        // Используем Array2 для эффективных операций
        // Используем view для избежания копирования данных
        let a_2d = a_arr
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert to 2D array");
        let b_2d = b_arr
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert to 2D array");
        
        // Используем встроенное матричное умножение ndarray (оптимизированное)
        // to_owned() создает копию только результата, а не входных данных
        let result = a_2d.dot(&b_2d);
        
        // Преобразуем обратно в ArrayD
        let result_d = result.into_dyn();
        Tensor::from_array(result_d)
    } else {
        panic!("matmul currently supports only 2D tensors");
    }
}

/// Транспонирование
pub fn transpose(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.t().to_owned();
    Tensor::from_array(result)
}

/// ReLU активация
pub fn relu(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| if x > 0.0 { x } else { 0.0 });
    Tensor::from_array(result)
}

/// LeakyReLU: x if x > 0 else alpha * x
pub fn leaky_relu(t: &Tensor, alpha: f32) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| if x > 0.0 { x } else { alpha * x });
    Tensor::from_array(result)
}

/// Sigmoid активация
pub fn sigmoid(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    Tensor::from_array(result)
}

/// Tanh активация
pub fn tanh(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| x.tanh());
    Tensor::from_array(result)
}

/// GELU (tanh approximation, PyTorch-compatible)
pub fn gelu(t: &Tensor) -> Tensor {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let a = 0.044715f32;
    let arr = t.data();
    let result = arr.mapv(|x| {
        let x3 = x * x * x;
        let inner = k * (x + a * x3);
        0.5 * x * (1.0 + inner.tanh())
    });
    Tensor::from_array(result)
}

/// d/dx GELU (tanh approximation)
pub fn gelu_derivative(t: &Tensor) -> Tensor {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let a = 0.044715f32;
    let arr = t.data();
    let result = arr.mapv(|x| {
        let x2 = x * x;
        let x3 = x * x2;
        let inner = k * (x + a * x3);
        let tnh = inner.tanh();
        let sech2 = 1.0 - tnh * tnh;
        let du_dx = k * (1.0 + 3.0 * a * x2);
        0.5 * (1.0 + tnh) + 0.5 * x * sech2 * du_dx
    });
    Tensor::from_array(result)
}

/// softplus(x) = ln(1 + exp(x))
pub fn softplus(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| (1.0 + x.exp()).ln());
    Tensor::from_array(result)
}

/// ELU: x if x > 0 else alpha * (exp(x) - 1)
pub fn elu(t: &Tensor, alpha: f32) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| {
        if x > 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    });
    Tensor::from_array(result)
}

/// SELU constants (Self-Normalizing Networks)
pub const SELU_ALPHA: f32 = 1.6732632423543772848170429916717;
pub const SELU_SCALE: f32 = 1.0507009873554804934193349852946;

pub fn selu(t: &Tensor) -> Tensor {
    let arr = t.data();
    let result = arr.mapv(|x| {
        if x > 0.0 {
            SELU_SCALE * x
        } else {
            SELU_SCALE * SELU_ALPHA * (x.exp() - 1.0)
        }
    });
    Tensor::from_array(result)
}

/// Softmax (правильная реализация по оси) — 2D: axis 0 или 1
pub fn softmax(t: &Tensor, axis: usize) -> Tensor {
    let arr = t.data();
    let shape = t.shape();

    if shape.len() == 2 && (axis == 0 || axis == 1) {
        let rows = shape[0];
        let cols = shape[1];
        let mut result_data = vec![0.0f32; rows * cols];

        if axis == 1 {
            for i in 0..rows {
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..cols {
                    max_val = max_val.max(arr[[i, j]]);
                }
                let mut exp_sum = 0.0f32;
                let mut exp_vals = vec![0.0f32; cols];
                for j in 0..cols {
                    let e = (arr[[i, j]] - max_val).exp();
                    exp_vals[j] = e;
                    exp_sum += e;
                }
                for j in 0..cols {
                    result_data[i * cols + j] = exp_vals[j] / exp_sum;
                }
            }
        } else {
            // axis == 0: softmax over rows for each column
            for j in 0..cols {
                let mut max_val = f32::NEG_INFINITY;
                for i in 0..rows {
                    max_val = max_val.max(arr[[i, j]]);
                }
                let mut exp_sum = 0.0f32;
                let mut exp_vals = vec![0.0f32; rows];
                for i in 0..rows {
                    let e = (arr[[i, j]] - max_val).exp();
                    exp_vals[i] = e;
                    exp_sum += e;
                }
                for i in 0..rows {
                    result_data[i * cols + j] = exp_vals[i] / exp_sum;
                }
            }
        }
        Tensor::from_slice(&result_data, shape)
    } else {
        // Fallback для других случаев - вычисляем по всему тензору
        let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_arr = arr.mapv(|x| (x - max_val).exp());
        let sum: f32 = exp_arr.sum();
        let result = exp_arr.mapv(|x| x / sum);
        Tensor::from_array(result)
    }
}

/// log(softmax(x, axis)) для 2D, axis 0 или 1 (численно стабильно)
pub fn log_softmax(t: &Tensor, axis: usize) -> Tensor {
    let sm = softmax(t, axis);
    let arr = sm.data();
    let result = arr.mapv(|x| x.max(1e-20).ln());
    Tensor::from_array(result)
}

/// Сумма по оси для 2D [rows, cols], axis 0 -> [1, cols], axis 1 -> [rows, 1] (keepdim broadcast-friendly)
pub fn sum_axis2d_keepdim(t: &Tensor, axis: usize) -> Tensor {
    let arr = t.data();
    let shape = t.shape();
    assert_eq!(shape.len(), 2);
    let rows = shape[0];
    let cols = shape[1];
    if axis == 1 {
        let mut out = vec![0.0f32; rows];
        for i in 0..rows {
            let mut s = 0.0f32;
            for j in 0..cols {
                s += arr[[i, j]];
            }
            out[i] = s;
        }
        Tensor::from_slice(&out, &[rows, 1])
    } else {
        let mut out = vec![0.0f32; cols];
        for j in 0..cols {
            let mut s = 0.0f32;
            for i in 0..rows {
                s += arr[[i, j]];
            }
            out[j] = s;
        }
        Tensor::from_slice(&out, &[1, cols])
    }
}

/// Сумма по последней оси с `keepdim` (последняя размерность становится 1).
/// 2D [rows, cols] → [rows, 1]; 1D [n] → [1, 1] (одна скалярная сумма).
pub fn sum_last_axis_keepdim(t: &Tensor) -> Tensor {
    let arr = t.data();
    let nd = arr.ndim();
    assert!(nd >= 1);
    if nd == 1 {
        let s: f32 = arr.iter().copied().sum::<f32>();
        return Tensor::from_slice(&[s], &[1, 1]);
    }
    let summed = arr.sum_axis(ndarray::Axis(nd - 1));
    let expanded = summed.insert_axis(ndarray::Axis(nd - 1));
    Tensor::from_array(expanded)
}

/// Среднее по последней оси с `keepdim` (деление на длину этой оси).
pub fn mean_last_axis_keepdim(t: &Tensor) -> Tensor {
    let arr = t.data();
    let nd = arr.ndim();
    assert!(nd >= 1);
    let n = arr.shape()[nd - 1] as f32;
    if nd == 1 {
        let s: f32 = arr.iter().copied().sum::<f32>() / n;
        return Tensor::from_slice(&[s], &[1, 1]);
    }
    let sum_t = sum_last_axis_keepdim(t);
    let a = sum_t.data();
    Tensor::from_array(a.mapv(|x| x / n))
}

/// Градиент softmax по входу (2D, axis 0 или 1): sm * (g - sum(sm*g, axis))
pub fn softmax_backward(grad_out: &Tensor, softmax_out: &Tensor, axis: usize) -> Tensor {
    let g = grad_out.data();
    let sm = softmax_out.data();
    let shape = softmax_out.shape();
    assert_eq!(shape.len(), 2);
    let rows = shape[0];
    let cols = shape[1];
    let mut result = vec![0.0f32; rows * cols];
    if axis == 1 {
        for i in 0..rows {
            let mut dot = 0.0f32;
            for j in 0..cols {
                dot += g[[i, j]] * sm[[i, j]];
            }
            for j in 0..cols {
                result[i * cols + j] = sm[[i, j]] * (g[[i, j]] - dot);
            }
        }
    } else {
        for j in 0..cols {
            let mut dot = 0.0f32;
            for i in 0..rows {
                dot += g[[i, j]] * sm[[i, j]];
            }
            for i in 0..rows {
                result[i * cols + j] = sm[[i, j]] * (g[[i, j]] - dot);
            }
        }
    }
    Tensor::from_slice(&result, shape)
}

/// Градиент log_softmax: g - sm * sum(g, axis)
pub fn log_softmax_backward(grad_out: &Tensor, softmax_out: &Tensor, axis: usize) -> Tensor {
    let g = grad_out.data();
    let sm = softmax_out.data();
    let shape = softmax_out.shape();
    assert_eq!(shape.len(), 2);
    let rows = shape[0];
    let cols = shape[1];
    let mut result = vec![0.0f32; rows * cols];
    if axis == 1 {
        for i in 0..rows {
            let mut s = 0.0f32;
            for j in 0..cols {
                s += g[[i, j]];
            }
            for j in 0..cols {
                result[i * cols + j] = g[[i, j]] - sm[[i, j]] * s;
            }
        }
    } else {
        for j in 0..cols {
            let mut s = 0.0f32;
            for i in 0..rows {
                s += g[[i, j]];
            }
            for i in 0..rows {
                result[i * cols + j] = g[[i, j]] - sm[[i, j]] * s;
            }
        }
    }
    Tensor::from_slice(&result, shape)
}

/// MaxPool2d [N,C,H,W], valid padding. Возвращает (output, argmax) где argmax той же формы что out, значения — плоский индекс во входе.
pub fn max_pool2d_forward(
    x: &Tensor,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
) -> (Tensor, Tensor) {
    let xs = x.shape();
    assert_eq!(xs.len(), 4);
    let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let h_out = (h - kh) / sy + 1;
    let w_out = (w - kw) / sx + 1;
    let mut out = vec![0.0f32; n * c * h_out * w_out];
    let mut argmax = vec![0.0f32; n * c * h_out * w_out];
    let xd = x.data();
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut best = f32::NEG_INFINITY;
                    let mut best_idx = 0usize;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sy + ki;
                            let wi = wo * sx + kj;
                            let v = xd[[ni, ci, hi, wi]];
                            if v > best {
                                best = v;
                                best_idx = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
                            }
                        }
                    }
                    let oi = ni * (c * h_out * w_out) + ci * (h_out * w_out) + ho * w_out + wo;
                    out[oi] = best;
                    argmax[oi] = best_idx as f32;
                }
            }
        }
    }
    (
        Tensor::from_slice(&out, &[n, c, h_out, w_out]),
        Tensor::from_slice(&argmax, &[n, c, h_out, w_out]),
    )
}

pub fn max_pool2d_backward(grad_out: &Tensor, argmax: &Tensor, input_shape: &[usize]) -> Tensor {
    let n_el: usize = input_shape.iter().product();
    let mut gi = vec![0.0f32; n_el];
    let gd = grad_out.data();
    let ad = argmax.data();
    let gs = grad_out.shape();
    let (nb, c, ho, wo) = (gs[0], gs[1], gs[2], gs[3]);
    for ni in 0..nb {
        for ci in 0..c {
            for y in 0..ho {
                for x in 0..wo {
                    let g = gd[[ni, ci, y, x]];
                    let idx = ad[[ni, ci, y, x]] as usize;
                    if idx < n_el {
                        gi[idx] += g;
                    }
                }
            }
        }
    }
    Tensor::from_slice(&gi, input_shape)
}

/// MaxPool1d [N,C,L], valid window.
pub fn max_pool1d_forward(
    x: &Tensor,
    k: usize,
    stride: usize,
) -> (Tensor, Tensor) {
    let xs = x.shape();
    assert_eq!(xs.len(), 3);
    let (n, c, l) = (xs[0], xs[1], xs[2]);
    let l_out = (l - k) / stride + 1;
    let mut out = vec![0.0f32; n * c * l_out];
    let mut argmax = vec![0.0f32; n * c * l_out];
    let xd = x.data();
    for ni in 0..n {
        for ci in 0..c {
            for lo in 0..l_out {
                let mut best = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for ki in 0..k {
                    let li = lo * stride + ki;
                    let v = xd[[ni, ci, li]];
                    if v > best {
                        best = v;
                        best_idx = ni * (c * l) + ci * l + li;
                    }
                }
                let oi = ni * (c * l_out) + ci * l_out + lo;
                out[oi] = best;
                argmax[oi] = best_idx as f32;
            }
        }
    }
    (
        Tensor::from_slice(&out, &[n, c, l_out]),
        Tensor::from_slice(&argmax, &[n, c, l_out]),
    )
}

pub fn max_pool1d_backward(grad_out: &Tensor, argmax: &Tensor, input_shape: &[usize]) -> Tensor {
    let n_el: usize = input_shape.iter().product();
    let mut gi = vec![0.0f32; n_el];
    let gd = grad_out.data();
    let ad = argmax.data();
    let gs = grad_out.shape();
    let (nb, c, lo) = (gs[0], gs[1], gs[2]);
    for ni in 0..nb {
        for ci in 0..c {
            for x in 0..lo {
                let g = gd[[ni, ci, x]];
                let idx = ad[[ni, ci, x]] as usize;
                if idx < n_el {
                    gi[idx] += g;
                }
            }
        }
    }
    Tensor::from_slice(&gi, input_shape)
}

/// AvgPool1d [N,C,L], valid window; output = sum / (k) over window.
pub fn avg_pool1d_forward(x: &Tensor, k: usize, stride: usize) -> Tensor {
    let xs = x.shape();
    assert_eq!(xs.len(), 3);
    let (n, c, l) = (xs[0], xs[1], xs[2]);
    let l_out = (l - k) / stride + 1;
    let scale = 1.0f32 / (k as f32);
    let mut out = vec![0.0f32; n * c * l_out];
    let xd = x.data();
    for ni in 0..n {
        for ci in 0..c {
            for lo in 0..l_out {
                let mut s = 0.0f32;
                for ki in 0..k {
                    let li = lo * stride + ki;
                    s += xd[[ni, ci, li]];
                }
                let oi = ni * (c * l_out) + ci * l_out + lo;
                out[oi] = s * scale;
            }
        }
    }
    Tensor::from_slice(&out, &[n, c, l_out])
}

pub fn avg_pool1d_backward(
    grad_out: &Tensor,
    input_shape: &[usize],
    k: usize,
    stride: usize,
) -> Tensor {
    let n = input_shape[0];
    let c = input_shape[1];
    let l_in = input_shape[2];
    let gs = grad_out.shape();
    let l_out = gs[2];
    let scale = 1.0f32 / (k as f32);
    let mut gi = vec![0.0f32; n * c * l_in];
    let gd = grad_out.data();
    for ni in 0..n {
        for ci in 0..c {
            for lo in 0..l_out {
                let g = gd[[ni, ci, lo]] * scale;
                for ki in 0..k {
                    let li = lo * stride + ki;
                    let ix = ni * (c * l_in) + ci * l_in + li;
                    gi[ix] += g;
                }
            }
        }
    }
    Tensor::from_slice(&gi, input_shape)
}

/// AvgPool2d [N,C,H,W], valid window.
pub fn avg_pool2d_forward(
    x: &Tensor,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
) -> Tensor {
    let xs = x.shape();
    assert_eq!(xs.len(), 4);
    let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let h_out = (h - kh) / sy + 1;
    let w_out = (w - kw) / sx + 1;
    let scale = 1.0f32 / ((kh * kw) as f32);
    let mut out = vec![0.0f32; n * c * h_out * w_out];
    let xd = x.data();
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut s = 0.0f32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sy + ki;
                            let wi = wo * sx + kj;
                            s += xd[[ni, ci, hi, wi]];
                        }
                    }
                    let oi = ni * (c * h_out * w_out) + ci * (h_out * w_out) + ho * w_out + wo;
                    out[oi] = s * scale;
                }
            }
        }
    }
    Tensor::from_slice(&out, &[n, c, h_out, w_out])
}

pub fn avg_pool2d_backward(
    grad_out: &Tensor,
    input_shape: &[usize],
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
) -> Tensor {
    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let win = input_shape[3];
    let gs = grad_out.shape();
    let h_out = gs[2];
    let w_out = gs[3];
    let scale = 1.0f32 / ((kh * kw) as f32);
    let mut gi = vec![0.0f32; n * c * h * win];
    let gd = grad_out.data();
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let g = gd[[ni, ci, ho, wo]] * scale;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sy + ki;
                            let wi = wo * sx + kj;
                            let ix = ni * (c * h * win) + ci * (h * win) + hi * win + wi;
                            gi[ix] += g;
                        }
                    }
                }
            }
        }
    }
    Tensor::from_slice(&gi, input_shape)
}

/// Global max over H,W: [N,C,H,W] -> [N,C,1,1]
pub fn global_max_pool2d_forward(x: &Tensor) -> (Tensor, Tensor) {
    let xs = x.shape();
    assert_eq!(xs.len(), 4);
    let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let mut out = vec![0.0f32; n * c];
    let mut argmax = vec![0.0f32; n * c];
    let xd = x.data();
    for ni in 0..n {
        for ci in 0..c {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for hi in 0..h {
                for wi in 0..w {
                    let v = xd[[ni, ci, hi, wi]];
                    if v > best {
                        best = v;
                        best_idx = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
                    }
                }
            }
            let oi = ni * c + ci;
            out[oi] = best;
            argmax[oi] = best_idx as f32;
        }
    }
    let out_t = Tensor::from_slice(&out, &[n, c, 1, 1]);
    let arg_t = Tensor::from_slice(&argmax, &[n, c, 1, 1]);
    (out_t, arg_t)
}

pub fn global_max_pool2d_backward(
    grad_out: &Tensor,
    argmax: &Tensor,
    input_shape: &[usize],
) -> Tensor {
    let n_el: usize = input_shape.iter().product();
    let mut gi = vec![0.0f32; n_el];
    let gd = grad_out.data();
    let ad = argmax.data();
    let gs = grad_out.shape();
    let (nb, c, _, _) = (gs[0], gs[1], gs[2], gs[3]);
    for ni in 0..nb {
        for ci in 0..c {
            let g = gd[[ni, ci, 0, 0]];
            let idx = ad[[ni, ci, 0, 0]] as usize;
            if idx < n_el {
                gi[idx] += g;
            }
        }
    }
    Tensor::from_slice(&gi, input_shape)
}

/// Global average over H,W: [N,C,H,W] -> [N,C,1,1]
pub fn global_avg_pool2d_forward(x: &Tensor) -> Tensor {
    let xs = x.shape();
    assert_eq!(xs.len(), 4);
    let (n, c, h, w) = (xs[0], xs[1], xs[2], xs[3]);
    let scale = 1.0f32 / ((h * w) as f32);
    let mut out = vec![0.0f32; n * c];
    let xd = x.data();
    for ni in 0..n {
        for ci in 0..c {
            let mut s = 0.0f32;
            for hi in 0..h {
                for wi in 0..w {
                    s += xd[[ni, ci, hi, wi]];
                }
            }
            out[ni * c + ci] = s * scale;
        }
    }
    Tensor::from_slice(&out, &[n, c, 1, 1])
}

pub fn global_avg_pool2d_backward(grad_out: &Tensor, input_shape: &[usize]) -> Tensor {
    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];
    let scale = 1.0f32 / ((h * w) as f32);
    let mut gi = vec![0.0f32; n * c * h * w];
    let gd = grad_out.data();
    for ni in 0..n {
        for ci in 0..c {
            let g = gd[[ni, ci, 0, 0]] * scale;
            for hi in 0..h {
                for wi in 0..w {
                    let ix = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
                    gi[ix] += g;
                }
            }
        }
    }
    Tensor::from_slice(&gi, input_shape)
}

/// Sum to shape (для broadcast градиентов)
pub fn sum_to_shape(t: &Tensor, target_shape: &[usize]) -> Result<Tensor, String> {
    let arr = t.data();
    let current_shape = t.shape();
    
    // Если формы уже совпадают, возвращаем как есть
    if current_shape == target_shape {
        return Ok(t.clone());
    }
    
    // Упрощенная реализация: reshape если возможно, иначе суммируем
    let total_elements: usize = target_shape.iter().product();
    if arr.len() == total_elements {
        // Просто reshape
        let reshaped = arr.clone().into_shape(ndarray::IxDyn(target_shape)).unwrap().to_owned();
        Ok(Tensor::from_array(reshaped))
    } else {
        // Нужно суммировать - упрощенная версия
        let mut result = arr.clone();
        // Суммируем по всем осям кроме последней
        while result.shape().len() > target_shape.len() {
            result = result.sum_axis(ndarray::Axis(0));
        }
        // Reshape к целевой форме
        if result.len() == total_elements {
            let reshaped = result.into_shape(ndarray::IxDyn(target_shape)).unwrap().to_owned();
            Ok(Tensor::from_array(reshaped))
        } else {
            Err(format!("Cannot sum from {:?} to {:?}", current_shape, target_shape))
        }
    }
}

