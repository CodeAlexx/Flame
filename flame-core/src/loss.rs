//! Loss functions with automatic differentiation support
//! 
//! This module provides common loss functions used in deep learning,
//! with full backward pass implementations for autograd.

use crate::{Tensor, Result, FlameError, AutogradContext, Op, TensorId};

/// Mean Squared Error (MSE) loss
/// 
/// Computes: mean((predictions - targets)^2)
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if predictions.shape != targets.shape {
        return Err(FlameError::ShapeMismatch {
            expected: predictions.shape.clone(),
            got: targets.shape.clone(),
        });
    }
    
    // Compute (predictions - targets)^2
    let diff = predictions.sub(targets)?;
    let squared = diff.square()?;
    
    // Mean over all elements
    let loss = squared.mean_all()?;
    
    // Record for autograd if needed
    if predictions.requires_grad || targets.requires_grad {
        let mut loss_with_grad = loss.clone()?;
        loss_with_grad.requires_grad = true;
        
        AutogradContext::record_op(
            loss_with_grad.id,
            Op::MSELoss { 
                predictions: predictions.id,
                targets: targets.id,
                num_elements: predictions.shape.elem_count()
            },
            vec![
                (predictions.id, predictions.clone()?),
                (targets.id, targets.clone()?)
            ]
        );
        
        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// L1 loss (Mean Absolute Error)
/// 
/// Computes: mean(|predictions - targets|)
pub fn l1_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if predictions.shape != targets.shape {
        return Err(FlameError::ShapeMismatch {
            expected: predictions.shape.clone(),
            got: targets.shape.clone(),
        });
    }
    
    // Compute |predictions - targets|
    let diff = predictions.sub(targets)?;
    let abs_diff = diff.abs()?;
    
    // Mean over all elements
    let loss = abs_diff.mean_all()?;
    
    // Record for autograd if needed
    if predictions.requires_grad || targets.requires_grad {
        let mut loss_with_grad = loss.clone()?;
        loss_with_grad.requires_grad = true;
        
        AutogradContext::record_op(
            loss_with_grad.id,
            Op::L1Loss {
                predictions: predictions.id,
                targets: targets.id,
                num_elements: predictions.shape.elem_count()
            },
            vec![
                (predictions.id, predictions.clone()?),
                (targets.id, targets.clone()?)
            ]
        );
        
        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Huber loss (Smooth L1 loss)
/// 
/// Combines advantages of MSE (smooth near zero) and L1 (robust to outliers)
/// loss = 0.5 * x^2                  if |x| <= delta
///      = delta * (|x| - 0.5 * delta) if |x| > delta
pub fn huber_loss(predictions: &Tensor, targets: &Tensor, delta: f32) -> Result<Tensor> {
    if predictions.shape != targets.shape {
        return Err(FlameError::ShapeMismatch {
            expected: predictions.shape.clone(),
            got: targets.shape.clone(),
        });
    }
    
    if delta <= 0.0 {
        return Err(FlameError::InvalidValue(
            format!("Huber loss delta must be positive, got {}", delta)
        ));
    }
    
    let diff = predictions.sub(targets)?;
    let abs_diff = diff.abs()?;
    
    // Create mask for |diff| <= delta
    let delta_tensor = Tensor::full(delta, diff.shape.clone(), diff.device.clone())?;
    let mask = abs_diff.le(&delta_tensor)?;
    
    // Quadratic part: 0.5 * diff^2
    let squared = diff.square()?.mul_scalar(0.5)?;
    
    // Linear part: delta * (|diff| - 0.5 * delta)
    let linear = abs_diff.sub_scalar(0.5 * delta)?.mul_scalar(delta)?;
    
    // Combine using mask
    let loss_elements = mask.where_tensor(&squared, &linear)?;
    
    // Mean over all elements
    let loss = loss_elements.mean_all()?;
    
    // Record for autograd if needed
    if predictions.requires_grad || targets.requires_grad {
        let mut loss_with_grad = loss.clone()?;
        loss_with_grad.requires_grad = true;
        
        AutogradContext::record_op(
            loss_with_grad.id,
            Op::HuberLoss {
                predictions: predictions.id,
                targets: targets.id,
                delta,
                num_elements: predictions.shape.elem_count()
            },
            vec![
                (predictions.id, predictions.clone()?),
                (targets.id, targets.clone()?)
            ]
        );
        
        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Binary Cross Entropy loss
/// 
/// Computes: -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions))
pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if predictions.shape != targets.shape {
        return Err(FlameError::ShapeMismatch {
            expected: predictions.shape.clone(),
            got: targets.shape.clone(),
        });
    }
    
    // Clamp predictions to avoid log(0)
    let eps = 1e-7;
    let predictions_clamped = predictions.clamp(eps, 1.0 - eps)?;
    
    // Compute BCE: -[y*log(p) + (1-y)*log(1-p)]
    let log_pred = predictions_clamped.log()?;
    let one_minus_pred = predictions_clamped.neg()?.add_scalar(1.0)?;
    let log_one_minus_pred = one_minus_pred.log()?;
    
    let one_minus_targets = targets.neg()?.add_scalar(1.0)?;
    
    let term1 = targets.mul(&log_pred)?;
    let term2 = one_minus_targets.mul(&log_one_minus_pred)?;
    
    let loss_elements = term1.add(&term2)?.neg()?;
    let loss = loss_elements.mean_all()?;
    
    // Record for autograd if needed
    if predictions.requires_grad || targets.requires_grad {
        let mut loss_with_grad = loss.clone()?;
        loss_with_grad.requires_grad = true;
        
        AutogradContext::record_op(
            loss_with_grad.id,
            Op::BCELoss {
                predictions: predictions.id,
                targets: targets.id,
                num_elements: predictions.shape.elem_count()
            },
            vec![
                (predictions.id, predictions.clone()?),
                (targets.id, targets.clone()?)
            ]
        );
        
        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

/// Cross Entropy loss for multi-class classification
/// 
/// Expects logits (unnormalized) as input, applies log_softmax internally
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // targets should be integer class indices
    if logits.shape.dims()[0] != targets.shape.dims()[0] {
        return Err(FlameError::InvalidOperation(
            "Batch size mismatch between logits and targets".into()
        ));
    }
    
    // Apply log_softmax to logits
    let log_probs = logits.log_softmax(-1)?;
    
    // Use negative log likelihood
    nll_loss(&log_probs, targets)
}

/// Negative Log Likelihood loss
/// 
/// Used with log_softmax output
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let batch_size = log_probs.shape.dims()[0];
    let num_classes = log_probs.shape.dims()[1];
    
    // Convert targets to indices if needed
    let target_data = targets.to_vec()?;
    let target_indices: Vec<i64> = target_data.iter().map(|&x| x as i64).collect();
    
    // Gather log probabilities for target classes
    let mut loss_sum = 0.0f32;
    let log_probs_data = log_probs.to_vec2::<f32>()?;
    
    for (i, &target) in target_indices.iter().enumerate() {
        if target < 0 || target >= num_classes as i64 {
            return Err(FlameError::InvalidValue(
                format!("Target index {} out of range [0, {})", target, num_classes)
            ));
        }
        loss_sum -= log_probs_data[i][target as usize];
    }
    
    let loss_val = loss_sum / batch_size as f32;
    let loss = Tensor::from_scalar(loss_val, log_probs.device.clone())?;
    
    // Record for autograd if needed
    if log_probs.requires_grad {
        let mut loss_with_grad = loss.clone()?;
        loss_with_grad.requires_grad = true;
        
        AutogradContext::record_op(
            loss_with_grad.id,
            Op::NLLLoss {
                log_probs: log_probs.id,
                targets: targets.id,
                batch_size
            },
            vec![
                (log_probs.id, log_probs.clone()?),
                (targets.id, targets.clone()?)
            ]
        );
        
        Ok(loss_with_grad)
    } else {
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    
    #[test]
    fn test_mse_loss() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        // Simple test case
        let predictions = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            device.clone()
        )?;
        let targets = Tensor::from_slice(
            &[1.5, 2.5, 3.5, 4.5],
            vec![2, 2],
            device.clone()
        )?;
        
        let loss = mse_loss(&predictions, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        
        // Expected: mean((0.5^2 + 0.5^2 + 0.5^2 + 0.5^2)) = 0.25
        assert!((loss_val - 0.25).abs() < 1e-6);
        
        Ok(())
    }
    
    #[test]
    fn test_l1_loss() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        let predictions = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            device.clone()
        )?;
        let targets = Tensor::from_slice(
            &[1.5, 2.5, 3.5, 4.5],
            vec![2, 2],
            device.clone()
        )?;
        
        let loss = l1_loss(&predictions, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        
        // Expected: mean(|0.5| + |0.5| + |0.5| + |0.5|) = 0.5
        assert!((loss_val - 0.5).abs() < 1e-6);
        
        Ok(())
    }
    
    #[test]
    fn test_huber_loss() -> Result<()> {
        let device = CudaDevice::new(0)?;
        
        // Mix of small and large differences
        let predictions = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 10.0],
            vec![4],
            device.clone()
        )?;
        let targets = Tensor::from_slice(
            &[1.2, 2.3, 3.0, 5.0],
            vec![4],
            device.clone()
        )?;
        
        let loss = huber_loss(&predictions, &targets, 1.0)?;
        
        // Verify loss is computed
        let _ = loss.to_scalar::<f32>()?;
        
        Ok(())
    }
}