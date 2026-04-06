// Autograd system for ML module
// Rewritten to match MetalNN architecture
// GradOp enum avoids Box<dyn Fn> in hot path (static dispatch in backward).

use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Gradient operation tag + captured data for backward. Static dispatch instead of Box<dyn Fn>.
#[derive(Clone)]
pub enum GradOp {
    Add {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    Mul {
        a_data: Tensor,
        b_data: Tensor,
    },
    Matmul {
        a_data: Tensor,
        b_data: Tensor,
    },
    Transpose,
    Relu {
        input_data: Tensor,
    },
    LeakyRelu {
        input_data: Tensor,
        alpha: f32,
    },
    Sigmoid {
        output_data: Tensor,
    },
    Tanh {
        output_data: Tensor,
    },
    Gelu {
        input_data: Tensor,
    },
    Softplus {
        input_data: Tensor,
    },
    Elu {
        input_data: Tensor,
        alpha: f32,
    },
    Selu {
        input_data: Tensor,
    },
    Softmax {
        softmax_out: Tensor,
        axis: usize,
    },
    LogSoftmax {
        softmax_out: Tensor,
        axis: usize,
    },
    Prelu {
        input_data: Tensor,
        alpha_scalar: f32,
    },
    Dropout {
        mask_scaled: Tensor,
    },
    Conv2d {
        input_data: Tensor,
        weight_data: Tensor,
        stride: (usize, usize),
        pad: (usize, usize),
        has_bias: bool,
    },
    MaxPool2d {
        argmax: Tensor,
        input_shape: Vec<usize>,
    },
    Conv1d {
        input_data: Tensor,
        weight_data: Tensor,
        stride: usize,
        pad: usize,
        has_bias: bool,
    },
    MaxPool1d {
        argmax: Tensor,
        input_shape: Vec<usize>,
    },
    AvgPool1d {
        input_shape: Vec<usize>,
        k: usize,
        stride: usize,
    },
    AvgPool2d {
        input_shape: Vec<usize>,
        kh: usize,
        kw: usize,
        sy: usize,
        sx: usize,
    },
    GlobalMaxPool2d {
        argmax: Tensor,
        input_shape: Vec<usize>,
    },
    GlobalAvgPool2d {
        input_shape: Vec<usize>,
    },
}

/// Legacy type for ABI/forward compatibility where a closure is still used externally.
pub type GradientFn = Box<dyn Fn(&Tensor) -> Vec<(usize, Tensor)>>;

/// Узел в computational graph
pub struct Variable {
    pub data: Rc<RefCell<Tensor>>,
    pub grad: Rc<RefCell<Option<Tensor>>>,
    /// Static op for backward; avoids RefCell<Option<Box<dyn Fn>>> in hot path.
    pub grad_op: RefCell<Option<GradOp>>,
    #[allow(dead_code)]
    pub grad_fn: RefCell<Option<GradientFn>>,
    pub parents: Vec<Rc<Variable>>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub id: usize,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn get_next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("id", &self.id)
            .finish()
    }
}

impl Variable {
    pub fn new(data: Tensor, requires_grad: bool) -> Rc<Self> {
        Rc::new(Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            grad_op: RefCell::new(None),
            grad_fn: RefCell::new(None),
            parents: vec![],
            requires_grad,
            is_leaf: true,
            id: get_next_id(),
        })
    }

    pub fn with_grad_fn(
        data: Tensor,
        requires_grad: bool,
        parents: Vec<Rc<Variable>>,
        grad_fn: GradientFn,
    ) -> Rc<Self> {
        Rc::new(Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            grad_op: RefCell::new(None),
            grad_fn: RefCell::new(Some(grad_fn)),
            parents,
            requires_grad,
            is_leaf: false,
            id: get_next_id(),
        })
    }

    /// Build variable with static GradOp (no Box<dyn Fn> in backward path).
    fn with_grad_op(
        data: Tensor,
        requires_grad: bool,
        parents: Vec<Rc<Variable>>,
        grad_op: GradOp,
    ) -> Rc<Self> {
        Rc::new(Self {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(None)),
            grad_op: RefCell::new(Some(grad_op)),
            grad_fn: RefCell::new(None),
            parents,
            requires_grad,
            is_leaf: false,
            id: get_next_id(),
        })
    }

    pub fn backward(&self, grad: Tensor) {
        if !self.requires_grad {
            return;
        }

        // Обновляем градиент
        {
            let mut current_grad = self.grad.borrow_mut();
            match *current_grad {
                Some(ref existing_grad) => {
                    let sum = add_grads(existing_grad, &grad);
                    *current_grad = Some(sum);
                }
                None => {
                    *current_grad = Some(grad.clone());
                }
            }
        }

        // Backward: static dispatch via GradOp first, then fallback to legacy grad_fn
        let grad_ref = self.grad.borrow();
        let g = match grad_ref.as_ref() {
            Some(gg) => gg,
            None => return,
        };

        if let Some(ref op) = *self.grad_op.borrow() {
            let parent_grads = grad_op_backward(op, g);
            drop(grad_ref);
            for (idx, parent_grad) in parent_grads {
                if idx < self.parents.len() {
                    self.parents[idx].backward(parent_grad);
                }
            }
            return;
        }

        if let Some(ref f) = *self.grad_fn.borrow() {
            let parent_grads = f(g);
            drop(grad_ref);
            for (idx, parent_grad) in parent_grads {
                if idx < self.parents.len() {
                    self.parents[idx].backward(parent_grad);
                }
            }
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }
}

fn add_grads(a: &Tensor, b: &Tensor) -> Tensor {
    use crate::ops::add;
    add(a, b)
}

/// Static backward dispatch for GradOp (no dyn call in hot path).
fn grad_op_backward(op: &GradOp, grad: &Tensor) -> Vec<(usize, Tensor)> {
    use crate::ops;
    match op {
        GradOp::Add { a_shape, b_shape } => {
            let grad_shape = grad.shape();
            let a_shape: &[usize] = a_shape;
            let b_shape: &[usize] = b_shape;
            if grad_shape == a_shape && grad_shape == b_shape {
                return vec![(0, grad.clone()), (1, grad.clone())];
            }
            let a_grad = if grad_shape == a_shape {
                grad.clone()
            } else {
                grad.clone()
            };
            let b_grad = if grad_shape == b_shape {
                grad.clone()
            } else if b_shape.len() == 1 && grad_shape.len() == 2 && b_shape[0] == grad_shape[1] {
                let grad_arr = grad.data();
                let mut summed = vec![0.0; b_shape[0]];
                for i in 0..grad_shape[0] {
                    for j in 0..b_shape[0] {
                        summed[j] += grad_arr[[i, j]];
                    }
                }
                Tensor::from_slice(&summed, b_shape)
            } else {
                grad.clone()
            };
            vec![(0, a_grad), (1, b_grad)]
        }
        GradOp::Mul { a_data, b_data } => {
            let a_grad = ops::mul(grad, b_data);
            let b_grad = ops::mul(grad, a_data);
            vec![(0, a_grad), (1, b_grad)]
        }
        GradOp::Matmul { a_data, b_data } => {
            let b_t = transpose(b_data);
            let a_t = transpose(a_data);
            let a_grad = ops::matmul(grad, &b_t);
            let b_grad = ops::matmul(&a_t, grad);
            vec![(0, a_grad), (1, b_grad)]
        }
        GradOp::Transpose => {
            let transposed_grad = transpose(grad);
            vec![(0, transposed_grad)]
        }
        GradOp::Relu { input_data } => {
            let arr = input_data.data();
            let mask = arr.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let mask_tensor = Tensor::from_array(mask);
            let input_grad = ops::mul(grad, &mask_tensor);
            vec![(0, input_grad)]
        }
        GradOp::LeakyRelu { input_data, alpha } => {
            let arr = input_data.data();
            let mask = arr.mapv(|x| if x > 0.0 { 1.0 } else { *alpha });
            let mask_tensor = Tensor::from_array(mask);
            let input_grad = ops::mul(grad, &mask_tensor);
            vec![(0, input_grad)]
        }
        GradOp::Sigmoid { output_data } => {
            let o = output_data.data();
            let local = o.mapv(|x| x * (1.0 - x));
            let local_t = Tensor::from_array(local);
            let input_grad = ops::mul(grad, &local_t);
            vec![(0, input_grad)]
        }
        GradOp::Tanh { output_data } => {
            let o = output_data.data();
            let local = o.mapv(|x| 1.0 - x * x);
            let local_t = Tensor::from_array(local);
            let input_grad = ops::mul(grad, &local_t);
            vec![(0, input_grad)]
        }
        GradOp::Gelu { input_data } => {
            let local = ops::gelu_derivative(input_data);
            let input_grad = ops::mul(grad, &local);
            vec![(0, input_grad)]
        }
        GradOp::Softplus { input_data } => {
            let sig = ops::sigmoid(input_data);
            let input_grad = ops::mul(grad, &sig);
            vec![(0, input_grad)]
        }
        GradOp::Elu { input_data, alpha } => {
            let arr = input_data.data();
            let g = grad.data();
            let mut grad_x = arr.mapv(|_| 0.0f32);
            for ((gx, &x), &gi) in grad_x.iter_mut().zip(arr.iter()).zip(g.iter()) {
                *gx = if x > 0.0 {
                    gi
                } else {
                    gi * alpha * x.exp()
                };
            }
            vec![(0, Tensor::from_array(grad_x))]
        }
        GradOp::Selu { input_data } => {
            let arr = input_data.data();
            let g = grad.data();
            let a = ops::SELU_ALPHA;
            let s = ops::SELU_SCALE;
            let mut grad_x = arr.mapv(|_| 0.0f32);
            for ((gx, &x), &gi) in grad_x.iter_mut().zip(arr.iter()).zip(g.iter()) {
                *gx = if x > 0.0 {
                    gi * s
                } else {
                    gi * s * a * x.exp()
                };
            }
            vec![(0, Tensor::from_array(grad_x))]
        }
        GradOp::Softmax { softmax_out, axis } => {
            let input_grad = ops::softmax_backward(grad, softmax_out, *axis);
            vec![(0, input_grad)]
        }
        GradOp::LogSoftmax { softmax_out, axis } => {
            let input_grad = ops::log_softmax_backward(grad, softmax_out, *axis);
            vec![(0, input_grad)]
        }
        GradOp::Prelu {
            input_data,
            alpha_scalar,
        } => {
            let arr = input_data.data();
            let g = grad.data();
            let mut gx = arr.clone();
            let mut ga = 0.0f32;
            for ((ix, &gi), x) in gx.iter_mut().zip(g.iter()).zip(arr.iter()) {
                let xv = *x;
                if xv > 0.0 {
                    *ix = gi;
                } else {
                    *ix = gi * alpha_scalar;
                    ga += gi * xv;
                }
            }
            vec![(0, Tensor::from_array(gx)), (1, Tensor::from_slice(&[ga], &[1]))]
        }
        GradOp::Dropout { mask_scaled } => {
            vec![(0, ops::mul(grad, mask_scaled))]
        }
        GradOp::Conv2d {
            input_data,
            weight_data,
            stride,
            pad,
            has_bias,
        } => {
            use crate::conv::{conv2d_grad_bias, conv2d_grad_input, conv2d_grad_weight};
            let gi = conv2d_grad_input(grad, weight_data, input_data.shape(), *stride, *pad);
            let gw = conv2d_grad_weight(grad, input_data, weight_data.shape(), *stride, *pad);
            if *has_bias {
                let gb = conv2d_grad_bias(grad, weight_data.shape()[0]);
                vec![(0, gi), (1, gw), (2, gb)]
            } else {
                vec![(0, gi), (1, gw)]
            }
        }
        GradOp::MaxPool2d {
            argmax,
            input_shape,
        } => {
            vec![(0, ops::max_pool2d_backward(grad, argmax, input_shape))]
        }
        GradOp::Conv1d {
            input_data,
            weight_data,
            stride,
            pad,
            has_bias,
        } => {
            use crate::conv::{
                conv1d_grad_bias, conv1d_grad_input, conv1d_grad_weight,
            };
            let gi = conv1d_grad_input(grad, weight_data, input_data.shape(), *stride, *pad);
            let gw = conv1d_grad_weight(grad, input_data, weight_data.shape(), *stride, *pad);
            if *has_bias {
                let gb = conv1d_grad_bias(grad, weight_data.shape()[0]);
                vec![(0, gi), (1, gw), (2, gb)]
            } else {
                vec![(0, gi), (1, gw)]
            }
        }
        GradOp::MaxPool1d {
            argmax,
            input_shape,
        } => {
            vec![(0, ops::max_pool1d_backward(grad, argmax, input_shape))]
        }
        GradOp::AvgPool1d {
            input_shape,
            k,
            stride,
        } => {
            vec![(0, ops::avg_pool1d_backward(grad, input_shape, *k, *stride))]
        }
        GradOp::AvgPool2d {
            input_shape,
            kh,
            kw,
            sy,
            sx,
        } => {
            vec![(0, ops::avg_pool2d_backward(
                grad, input_shape, *kh, *kw, *sy, *sx,
            ))]
        }
        GradOp::GlobalMaxPool2d {
            argmax,
            input_shape,
        } => {
            vec![(0, ops::global_max_pool2d_backward(grad, argmax, input_shape))]
        }
        GradOp::GlobalAvgPool2d { input_shape } => {
            vec![(0, ops::global_avg_pool2d_backward(grad, input_shape))]
        }
    }
}

/// Создать Variable с requires_grad=true
pub fn requires_grad(t: Tensor) -> Rc<Variable> {
    Variable::new(t, true)
}

/// Операции с autograd

use crate::ops;

/// Сложение с autograd
pub fn add_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    let a_data_ref = a.data.borrow();
    let b_data_ref = b.data.borrow();
    let result_data = ops::add(&a_data_ref, &b_data_ref);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let a_shape = a_data_ref.shape().to_vec();
    let b_shape = b_data_ref.shape().to_vec();
    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_op = GradOp::Add {
        a_shape,
        b_shape,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// Умножение с autograd
pub fn mul_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    let a_data_ref = a.data.borrow();
    let b_data_ref = b.data.borrow();
    let result_data = ops::mul(&a_data_ref, &b_data_ref);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_op = GradOp::Mul {
        a_data: a_data_ref.clone(),
        b_data: b_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// Матричное умножение с autograd
pub fn matmul_with_grad(a: Rc<Variable>, b: Rc<Variable>) -> Rc<Variable> {
    let a_cpu = a.data.borrow().to_cpu()
        .map_err(|e| format!("Failed to convert tensor a to CPU: {}", e))
        .unwrap_or_else(|_| a.data.borrow().clone());
    let b_cpu = b.data.borrow().to_cpu()
        .map_err(|e| format!("Failed to convert tensor b to CPU: {}", e))
        .unwrap_or_else(|_| b.data.borrow().clone());

    let result_data = ops::matmul(&a_cpu, &b_cpu);

    let requires_grad = a.requires_grad || b.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&a), Rc::clone(&b)];
    let grad_op = GradOp::Matmul {
        a_data: a_cpu.clone(),
        b_data: b_cpu.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

fn transpose(t: &Tensor) -> Tensor {
    // Convert to CPU if tensor is on GPU (ops require CPU tensors)
    let t_cpu = match t.to_cpu() {
        Ok(cpu_t) => cpu_t,
        Err(_e) => t.clone(),
    };
    
    // Validate tensor shape before transpose
    let shape = t_cpu.shape();
    if shape.len() != 2 {
        // For non-2D tensors, return as-is (transpose only works for 2D)
        return t_cpu;
    }
    
    // Use ops::transpose which handles 2D tensors correctly
    // This ensures proper handling of the tensor data
    crate::ops::transpose(&t_cpu)
}

/// Транспонирование с autograd
pub fn transpose_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = transpose(&input_data_ref);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Transpose;
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// ReLU с autograd
pub fn relu_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::relu(&input_data_ref);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Relu {
        input_data: input_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// LeakyReLU с autograd
pub fn leaky_relu_with_grad(input: Rc<Variable>, alpha: f32) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::leaky_relu(&input_data_ref, alpha);

    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }

    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::LeakyRelu {
        input_data: input_data_ref.clone(),
        alpha,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn sigmoid_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::sigmoid(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Sigmoid {
        output_data: result_data.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn tanh_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::tanh(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Tanh {
        output_data: result_data.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn gelu_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::gelu(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Gelu {
        input_data: input_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn softplus_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::softplus(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Softplus {
        input_data: input_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn elu_with_grad(input: Rc<Variable>, alpha: f32) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::elu(&input_data_ref, alpha);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Elu {
        input_data: input_data_ref.clone(),
        alpha,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn selu_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::selu(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Selu {
        input_data: input_data_ref.clone(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// Softmax по оси (только 2D тензоры, axis 0 или 1)
pub fn softmax_with_grad(input: Rc<Variable>, axis: usize) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let result_data = ops::softmax(&input_data_ref, axis);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Softmax {
        softmax_out: result_data.clone(),
        axis,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn log_softmax_with_grad(input: Rc<Variable>, axis: usize) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let sm = ops::softmax(&input_data_ref, axis);
    let result_data = ops::log_softmax(&input_data_ref, axis);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::LogSoftmax {
        softmax_out: sm,
        axis,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// PReLU: y = x if x > 0 else alpha * x; alpha — скаляр [1]
mod dropout_rng {
    use std::sync::atomic::{AtomicU64, Ordering};
    static S: AtomicU64 = AtomicU64::new(1);
    pub fn rand01() -> f32 {
        let x = S.fetch_add(1, Ordering::Relaxed).wrapping_mul(1103515245).wrapping_add(12345);
        ((x >> 16) & 0xFFFF) as f32 / 65536.0
    }
}

/// Inverted dropout: training uses mask / (1-p); eval is identity.
pub fn dropout_with_grad(input: Rc<Variable>, p: f32) -> Rc<Variable> {
    use crate::forward_mode::forward_training;
    if !forward_training() || p <= 0.0 {
        return input;
    }
    if p >= 1.0 {
        let shape = input.data.borrow().shape().to_vec();
        return Variable::new(Tensor::zeros(shape), input.requires_grad);
    }
    let input_data_ref = input.data.borrow();
    let shape = input_data_ref.shape().to_vec();
    let n: usize = shape.iter().product();
    let scale = 1.0 / (1.0 - p);
    let mut mask = vec![0.0f32; n];
    for i in 0..n {
        mask[i] = if dropout_rng::rand01() > p {
            scale
        } else {
            0.0
        };
    }
    let mask_t = Tensor::from_slice(&mask, &shape);
    let result_data = ops::mul(&input_data_ref, &mask_t);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Dropout {
        mask_scaled: mask_t,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

/// Spatial dropout: same mask per channel map [N,C,H,W].
pub fn dropout2d_channel_with_grad(input: Rc<Variable>, p: f32) -> Rc<Variable> {
    use crate::forward_mode::forward_training;
    if !forward_training() || p <= 0.0 {
        return input;
    }
    if p >= 1.0 {
        let shape = input.data.borrow().shape().to_vec();
        return Variable::new(Tensor::zeros(shape), input.requires_grad);
    }
    let ndim = input.data.borrow().shape().len();
    if ndim != 4 {
        return dropout_with_grad(input, p);
    }
    let input_data_ref = input.data.borrow();
    let shape = input_data_ref.shape().to_vec();
    let n = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    let scale = 1.0 / (1.0 - p);
    let mut mask = vec![0.0f32; n * c * h * w];
    for ni in 0..n {
        for ci in 0..c {
            let v = if dropout_rng::rand01() > p {
                scale
            } else {
                0.0
            };
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
                    mask[idx] = v;
                }
            }
        }
    }
    let mask_t = Tensor::from_slice(&mask, &shape);
    let result_data = ops::mul(&input_data_ref, &mask_t);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::Dropout {
        mask_scaled: mask_t,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn max_pool2d_with_grad(
    input: Rc<Variable>,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let input_shape = input_data_ref.shape().to_vec();
    let (result_data, argmax) = ops::max_pool2d_forward(&input_data_ref, kh, kw, sy, sx);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::MaxPool2d {
        argmax,
        input_shape,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn prelu_with_grad(input: Rc<Variable>, alpha: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let alpha_data_ref = alpha.data.borrow();
    let a = alpha_data_ref.data().iter().next().copied().unwrap_or(0.25);
    let arr = input_data_ref.data();
    let result = arr.mapv(|x| if x > 0.0 { x } else { a * x });
    let result_data = Tensor::from_array(result);
    let requires_grad = input.requires_grad || alpha.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input), Rc::clone(&alpha)];
    let grad_op = GradOp::Prelu {
        input_data: input_data_ref.clone(),
        alpha_scalar: a,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn conv2d_with_grad(
    input: Rc<Variable>,
    weight: Rc<Variable>,
    bias: Option<Rc<Variable>>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let weight_data_ref = weight.data.borrow();
    let bias_ref = bias.as_ref().map(|b| b.data.borrow());
    let result_data = crate::conv::conv2d_forward(
        &input_data_ref,
        &weight_data_ref,
        match &bias_ref {
            Some(r) => Some(&**r),
            None => None,
        },
        stride,
        pad,
    );
    let requires_grad = input.requires_grad || weight.requires_grad || bias.as_ref().map(|b| b.requires_grad).unwrap_or(false);
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let mut parents = vec![Rc::clone(&input), Rc::clone(&weight)];
    if let Some(ref b) = bias {
        parents.push(Rc::clone(b));
    }
    let grad_op = GradOp::Conv2d {
        input_data: input_data_ref.clone(),
        weight_data: weight_data_ref.clone(),
        stride,
        pad,
        has_bias: bias.is_some(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn conv1d_with_grad(
    input: Rc<Variable>,
    weight: Rc<Variable>,
    bias: Option<Rc<Variable>>,
    stride: usize,
    pad: usize,
) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let weight_data_ref = weight.data.borrow();
    let bias_ref = bias.as_ref().map(|b| b.data.borrow());
    let result_data = crate::conv::conv1d_forward(
        &input_data_ref,
        &weight_data_ref,
        match &bias_ref {
            Some(r) => Some(&**r),
            None => None,
        },
        stride,
        pad,
    );
    let requires_grad = input.requires_grad
        || weight.requires_grad
        || bias.as_ref().map(|b| b.requires_grad).unwrap_or(false);
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let mut parents = vec![Rc::clone(&input), Rc::clone(&weight)];
    if let Some(ref b) = bias {
        parents.push(Rc::clone(b));
    }
    let grad_op = GradOp::Conv1d {
        input_data: input_data_ref.clone(),
        weight_data: weight_data_ref.clone(),
        stride,
        pad,
        has_bias: bias.is_some(),
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn max_pool1d_with_grad(
    input: Rc<Variable>,
    k: usize,
    stride: usize,
) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let input_shape = input_data_ref.shape().to_vec();
    let (result_data, argmax) = ops::max_pool1d_forward(&input_data_ref, k, stride);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::MaxPool1d {
        argmax,
        input_shape,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn avg_pool1d_with_grad(
    input: Rc<Variable>,
    k: usize,
    stride: usize,
) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let input_shape = input_data_ref.shape().to_vec();
    let result_data = ops::avg_pool1d_forward(&input_data_ref, k, stride);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::AvgPool1d {
        input_shape,
        k,
        stride,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn avg_pool2d_with_grad(
    input: Rc<Variable>,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let input_shape = input_data_ref.shape().to_vec();
    let result_data = ops::avg_pool2d_forward(&input_data_ref, kh, kw, sy, sx);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::AvgPool2d {
        input_shape,
        kh,
        kw,
        sy,
        sx,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn global_max_pool2d_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let input_shape = input_data_ref.shape().to_vec();
    let (result_data, argmax) = ops::global_max_pool2d_forward(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::GlobalMaxPool2d {
        argmax,
        input_shape,
    };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}

pub fn global_avg_pool2d_with_grad(input: Rc<Variable>) -> Rc<Variable> {
    let input_data_ref = input.data.borrow();
    let input_shape = input_data_ref.shape().to_vec();
    let result_data = ops::global_avg_pool2d_forward(&input_data_ref);
    let requires_grad = input.requires_grad;
    if !requires_grad {
        return Variable::new(result_data, false);
    }
    let parents = vec![Rc::clone(&input)];
    let grad_op = GradOp::GlobalAvgPool2d { input_shape };
    Variable::with_grad_op(result_data, requires_grad, parents, grad_op)
}
