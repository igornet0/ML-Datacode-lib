//! Naive CPU conv2d forward / backward for autograd (small tensors).

use crate::tensor::Tensor;

/// [N,C,H,W] conv2d, stride and padding, weight [outC,inC,kH,kW], bias [outC] optional
pub fn conv2d_forward(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Tensor {
    let xs = x.shape();
    let ws = w.shape();
    assert_eq!(xs.len(), 4, "conv2d input must be 4D");
    assert_eq!(ws.len(), 4, "conv2d weight must be 4D");
    let (n, cin, h, w_in) = (xs[0], xs[1], xs[2], xs[3]);
    let (cout, cin_w, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    assert_eq!(cin, cin_w);
    let (sy, sx) = stride;
    let (py, px) = pad;
    let h_out = (h + 2 * py - kh) / sy + 1;
    let w_out = (w_in + 2 * px - kw) / sx + 1;
    let mut out = vec![0.0f32; n * cout * h_out * w_out];
    let xd = x.data();
    let wd = w.data();
    let bd = bias.map(|b| b.data());
    for ni in 0..n {
        for co in 0..cout {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut s = bd
                        .map(|b| b[[co]])
                        .unwrap_or(0.0);
                    for ci in 0..cin {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sy + ki;
                                let wi = wo * sx + kj;
                                let h_pad = hi as isize - py as isize;
                                let w_pad = wi as isize - px as isize;
                                if h_pad >= 0
                                    && w_pad >= 0
                                    && (h_pad as usize) < h
                                    && (w_pad as usize) < w_in
                                {
                                    let xv = xd[[ni, ci, h_pad as usize, w_pad as usize]];
                                    let wv = wd[[co, ci, ki, kj]];
                                    s += xv * wv;
                                }
                            }
                        }
                    }
                    let oi = ni * (cout * h_out * w_out) + co * (h_out * w_out) + ho * w_out + wo;
                    out[oi] = s;
                }
            }
        }
    }
    Tensor::from_slice(&out, &[n, cout, h_out, w_out])
}

pub fn conv2d_grad_input(
    grad_out: &Tensor,
    w: &Tensor,
    x_shape: &[usize],
    stride: (usize, usize),
    pad: (usize, usize),
) -> Tensor {
    let gs = grad_out.shape();
    let ws = w.shape();
    let (n, cout, ho, wo) = (gs[0], gs[1], gs[2], gs[3]);
    let (_, cin, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    let (h, win) = (x_shape[2], x_shape[3]);
    let (sy, sx) = stride;
    let (py, px) = pad;
    let mut gx = vec![0.0f32; n * cin * h * win];
    let gd = grad_out.data();
    let wd = w.data();
    for ni in 0..n {
        for co in 0..cout {
            for y in 0..ho {
                for xo in 0..wo {
                    let g = gd[[ni, co, y, xo]];
                    for ci in 0..cin {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = y * sy + ki;
                                let wi = xo * sx + kj;
                                let h_pad = hi as isize - py as isize;
                                let w_pad = wi as isize - px as isize;
                                if h_pad >= 0
                                    && w_pad >= 0
                                    && (h_pad as usize) < h
                                    && (w_pad as usize) < win
                                {
                                    let ix = ni * (cin * h * win)
                                        + ci * (h * win)
                                        + (h_pad as usize) * win
                                        + (w_pad as usize);
                                    gx[ix] += g * wd[[co, ci, ki, kj]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_slice(&gx, x_shape)
}

pub fn conv2d_grad_weight(
    grad_out: &Tensor,
    x: &Tensor,
    w_shape: &[usize],
    stride: (usize, usize),
    pad: (usize, usize),
) -> Tensor {
    let gs = grad_out.shape();
    let xs = x.shape();
    let (n, cout, ho, wo_) = (gs[0], gs[1], gs[2], gs[3]);
    let (_, cin, h, win) = (xs[0], xs[1], xs[2], xs[3]);
    let (_, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    let (sy, sx) = stride;
    let (py, px) = pad;
    let mut gw = vec![0.0f32; w_shape.iter().product::<usize>()];
    let gd = grad_out.data();
    let xd = x.data();
    for ni in 0..n {
        for co in 0..cout {
            for y in 0..ho {
                for xo in 0..wo_ {
                    let g = gd[[ni, co, y, xo]];
                    for ci in 0..cin {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = y * sy + ki;
                                let wi = xo * sx + kj;
                                let h_pad = hi as isize - py as isize;
                                let w_pad = wi as isize - px as isize;
                                if h_pad >= 0
                                    && w_pad >= 0
                                    && (h_pad as usize) < h
                                    && (w_pad as usize) < win
                                {
                                    let xv = xd[[ni, ci, h_pad as usize, w_pad as usize]];
                                    let wi_ = co * (cin * kh * kw) + ci * (kh * kw) + ki * kw + kj;
                                    gw[wi_] += g * xv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_slice(&gw, w_shape)
}

pub fn conv2d_grad_bias(grad_out: &Tensor, cout: usize) -> Tensor {
    let gs = grad_out.shape();
    let n = gs[0];
    let ho = gs[2];
    let wo = gs[3];
    let mut b = vec![0.0f32; cout];
    let gd = grad_out.data();
    for co in 0..cout {
        let mut s = 0.0f32;
        for ni in 0..n {
            for y in 0..ho {
                for x in 0..wo {
                    s += gd[[ni, co, y, x]];
                }
            }
        }
        b[co] = s;
    }
    Tensor::from_slice(&b, &[cout])
}

// --- Conv1d: input [N, C_in, L], weight [C_out, C_in, K] ---

/// Naive conv1d forward.
pub fn conv1d_forward(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    pad: usize,
) -> Tensor {
    let xs = x.shape();
    let ws = w.shape();
    assert_eq!(xs.len(), 3, "conv1d input must be 3D [N,C,L]");
    assert_eq!(ws.len(), 3, "conv1d weight must be 3D [outC,inC,K]");
    let (n, cin, l_in) = (xs[0], xs[1], xs[2]);
    let (cout, cin_w, k) = (ws[0], ws[1], ws[2]);
    assert_eq!(cin, cin_w);
    let l_out = (l_in + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; n * cout * l_out];
    let xd = x.data();
    let wd = w.data();
    let bd = bias.map(|b| b.data());
    for ni in 0..n {
        for co in 0..cout {
            for lo in 0..l_out {
                let mut s = bd.map(|b| b[[co]]).unwrap_or(0.0);
                for ci in 0..cin {
                    for ki in 0..k {
                        let li = lo * stride + ki;
                        let l_pad = li as isize - pad as isize;
                        if l_pad >= 0 && (l_pad as usize) < l_in {
                            s += xd[[ni, ci, l_pad as usize]] * wd[[co, ci, ki]];
                        }
                    }
                }
                let oi = ni * (cout * l_out) + co * l_out + lo;
                out[oi] = s;
            }
        }
    }
    Tensor::from_slice(&out, &[n, cout, l_out])
}

pub fn conv1d_grad_input(
    grad_out: &Tensor,
    w: &Tensor,
    x_shape: &[usize],
    stride: usize,
    pad: usize,
) -> Tensor {
    let gs = grad_out.shape();
    let ws = w.shape();
    let (n, cout, lo) = (gs[0], gs[1], gs[2]);
    let (_, cin, k) = (ws[0], ws[1], ws[2]);
    let l_in = x_shape[2];
    let mut gx = vec![0.0f32; n * cin * l_in];
    let gd = grad_out.data();
    let wd = w.data();
    for ni in 0..n {
        for co in 0..cout {
            for xo in 0..lo {
                let g = gd[[ni, co, xo]];
                for ci in 0..cin {
                    for ki in 0..k {
                        let li = xo * stride + ki;
                        let l_pad = li as isize - pad as isize;
                        if l_pad >= 0 && (l_pad as usize) < l_in {
                            let ix = ni * (cin * l_in) + ci * l_in + (l_pad as usize);
                            gx[ix] += g * wd[[co, ci, ki]];
                        }
                    }
                }
            }
        }
    }
    Tensor::from_slice(&gx, x_shape)
}

pub fn conv1d_grad_weight(
    grad_out: &Tensor,
    x: &Tensor,
    w_shape: &[usize],
    stride: usize,
    pad: usize,
) -> Tensor {
    let gs = grad_out.shape();
    let xs = x.shape();
    let (n, cout, lo_) = (gs[0], gs[1], gs[2]);
    let (_, cin, l_in) = (xs[0], xs[1], xs[2]);
    let (_, _, k) = (w_shape[0], w_shape[1], w_shape[2]);
    let mut gw = vec![0.0f32; w_shape.iter().product::<usize>()];
    let gd = grad_out.data();
    let xd = x.data();
    for ni in 0..n {
        for co in 0..cout {
            for xo in 0..lo_ {
                let g = gd[[ni, co, xo]];
                for ci in 0..cin {
                    for ki in 0..k {
                        let li = xo * stride + ki;
                        let l_pad = li as isize - pad as isize;
                        if l_pad >= 0 && (l_pad as usize) < l_in {
                            let xv = xd[[ni, ci, l_pad as usize]];
                            let wi_ = co * (cin * k) + ci * k + ki;
                            gw[wi_] += g * xv;
                        }
                    }
                }
            }
        }
    }
    Tensor::from_slice(&gw, w_shape)
}

pub fn conv1d_grad_bias(grad_out: &Tensor, cout: usize) -> Tensor {
    let gs = grad_out.shape();
    let n = gs[0];
    let lo = gs[2];
    let mut b = vec![0.0f32; cout];
    let gd = grad_out.data();
    for co in 0..cout {
        let mut s = 0.0f32;
        for ni in 0..n {
            for x in 0..lo {
                s += gd[[ni, co, x]];
            }
        }
        b[co] = s;
    }
    Tensor::from_slice(&b, &[cout])
}
