//! CPU implementation of pooling operations

use crate::{Tensor, Shape, Result, FlameError};

/// MaxPool2d forward implementation
pub fn maxpool2d_forward(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(Tensor, Tensor)> {
    let shape = input.shape().dims();
    let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    
    // Calculate output dimensions
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    
    // Get input data
    let input_data = input.to_vec()?;
    let mut output_data = vec![f32::NEG_INFINITY; batch * channels * h_out * w_out];
    let mut indices_data = vec![0u32; batch * channels * h_out * w_out];
    
    // Perform max pooling
    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    
                    // Find max in kernel window
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0u32;
                    
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;
                            
                            if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                                let real_ih = ih - ph;
                                let real_iw = iw - pw;
                                
                                if real_ih < h_in && real_iw < w_in {
                                    let in_idx = ((b * channels + c) * h_in + real_ih) * w_in + real_iw;
                                    let val = input_data[in_idx];
                                    
                                    if val > max_val {
                                        max_val = val;
                                        max_idx = in_idx as u32;
                                    }
                                }
                            }
                        }
                    }
                    
                    output_data[out_idx] = max_val;
                    indices_data[out_idx] = max_idx;
                }
            }
        }
    }
    
    let output = Tensor::from_vec(
        output_data,
        Shape::from_dims(&[batch, channels, h_out, w_out]),
        input.device.clone()
    )?;
    
    let indices = Tensor::from_vec(
        indices_data.iter().map(|&x| x as f32).collect(),
        Shape::from_dims(&[batch, channels, h_out, w_out]),
        input.device.clone()
    )?;
    
    Ok((output, indices))
}

/// MaxPool2d backward implementation
pub fn maxpool2d_backward(
    grad_output: &Tensor,
    input_shape: Shape,
    indices: &Tensor,
) -> Result<Tensor> {
    let grad_output_data = grad_output.to_vec()?;
    let indices_data = indices.to_vec()?;
    
    let input_size = input_shape.elem_count();
    let mut grad_input_data = vec![0.0f32; input_size];
    
    // Scatter gradients back using indices
    for (i, &grad) in grad_output_data.iter().enumerate() {
        let idx = indices_data[i] as usize;
        if idx < input_size {
            grad_input_data[idx] += grad;
        }
    }
    
    Tensor::from_vec(grad_input_data, input_shape, grad_output.device.clone())
}

/// AvgPool2d forward implementation
pub fn avgpool2d_forward(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
) -> Result<Tensor> {
    let shape = input.shape().dims();
    let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    
    // Calculate output dimensions
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    
    // Get input data
    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; batch * channels * h_out * w_out];
    
    // Perform average pooling
    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    
                    let mut sum = 0.0f32;
                    let mut count = 0;
                    
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;
                            
                            if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                                let real_ih = ih - ph;
                                let real_iw = iw - pw;
                                
                                if real_ih < h_in && real_iw < w_in {
                                    let in_idx = ((b * channels + c) * h_in + real_ih) * w_in + real_iw;
                                    sum += input_data[in_idx];
                                    count += 1;
                                } else if count_include_pad {
                                    count += 1;
                                }
                            } else if count_include_pad {
                                count += 1;
                            }
                        }
                    }
                    
                    output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }
    
    Tensor::from_vec(
        output_data,
        Shape::from_dims(&[batch, channels, h_out, w_out]),
        input.device.clone()
    )
}

/// AvgPool2d backward implementation
pub fn avgpool2d_backward(
    grad_output: &Tensor,
    input_shape: Shape,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
) -> Result<Tensor> {
    let grad_output_data = grad_output.to_vec()?;
    let grad_shape = grad_output.shape().dims();
    
    let (batch, channels, h_in, w_in) = (
        input_shape.dims()[0],
        input_shape.dims()[1],
        input_shape.dims()[2],
        input_shape.dims()[3],
    );
    let (_, _, h_out, w_out) = (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    
    let mut grad_input_data = vec![0.0f32; batch * channels * h_in * w_in];
    
    // Scatter gradients back
    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    let grad = grad_output_data[out_idx];
                    
                    // Count elements in kernel window
                    let mut count = 0;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;
                            
                            if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                                let real_ih = ih - ph;
                                let real_iw = iw - pw;
                                
                                if real_ih < h_in && real_iw < w_in {
                                    count += 1;
                                } else if count_include_pad {
                                    count += 1;
                                }
                            } else if count_include_pad {
                                count += 1;
                            }
                        }
                    }
                    
                    // Distribute gradient
                    let grad_per_elem = if count > 0 { grad / count as f32 } else { 0.0 };
                    
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = oh * sh + kh_i;
                            let iw = ow * sw + kw_i;
                            
                            if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                                let real_ih = ih - ph;
                                let real_iw = iw - pw;
                                
                                if real_ih < h_in && real_iw < w_in {
                                    let in_idx = ((b * channels + c) * h_in + real_ih) * w_in + real_iw;
                                    grad_input_data[in_idx] += grad_per_elem;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Tensor::from_vec(grad_input_data, input_shape, grad_output.device.clone())
}