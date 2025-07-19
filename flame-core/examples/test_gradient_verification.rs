//! Comprehensive verification of FLAME gradient operations
//! These tests WILL FAIL if gradient operations are incorrect

use flame_core::{
    Tensor, Shape, Result, CudaDevice, AutogradContext, FlameError,
    cuda_gradient_ops::CudaGradientOps,
};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("🔬 FLAME Gradient Operation Verification\n");
    println!("These tests verify actual correctness, not just 'no crash'\n");
    
    let device = CudaDevice::new(0)?;
    let gpu_ops = CudaGradientOps::new(device.clone())?;
    
    // Test 1: Gradient Clipping MUST Actually Clip
    println!("Test 1: Verify Gradient Clipping Bounds");
    {
        // Create gradient with values outside clip range
        let test_values = vec![-10.0, -5.0, -1.5, -0.5, 0.0, 0.5, 1.5, 5.0, 10.0];
        let mut grad = Tensor::from_vec(
            test_values.clone(),
            Shape::from_dims(&[9]),
            device.clone()
        )?;
        
        // Apply clipping to [-1.0, 1.0]
        gpu_ops.clip_gradient(&mut grad, 1.0)?;
        
        // Verify ALL values are within bounds
        let clipped = grad.to_vec()?;
        for (i, &val) in clipped.iter().enumerate() {
            if val < -1.0 || val > 1.0 {
                panic!("❌ FAILED: Value {} at index {} is outside [-1, 1] after clipping!", val, i);
            }
            
            // Verify exact clipping behavior
            let original = test_values[i];
            let expected = original.clamp(-1.0, 1.0);
            if (val - expected).abs() > 1e-6 {
                panic!("❌ FAILED: Clipped value {} doesn't match expected {} for original {}", 
                       val, expected, original);
            }
        }
        println!("✅ Gradient clipping correctly enforces bounds");
    }
    
    // Test 2: L2 Normalization MUST Produce Correct Norm
    println!("\nTest 2: Verify L2 Normalization Accuracy");
    {
        // Test multiple gradient vectors with known norms
        let test_cases = vec![
            (vec![3.0, 4.0], 5.0),  // 3-4-5 triangle
            (vec![1.0, 0.0, 0.0], 1.0),  // unit vector
            (vec![2.0, 2.0, 2.0, 2.0], 4.0),  // sqrt(16) = 4
        ];
        
        for (values, original_norm) in test_cases {
            let mut grad = Tensor::from_vec(
                values.clone(),
                Shape::from_dims(&[values.len()]),
                device.clone()
            )?;
            
            // Normalize to max norm of 2.0
            let max_norm = 2.0;
            gpu_ops.normalize_gradient(&mut grad, max_norm)?;
            
            // Compute actual norm after normalization
            let normalized = grad.to_vec()?;
            let actual_norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            // If original norm > max_norm, it should be scaled down to max_norm
            // If original norm <= max_norm, it should remain unchanged
            let expected_norm = if original_norm > max_norm { max_norm } else { original_norm };
            
            if (actual_norm - expected_norm).abs() > 1e-5 {
                panic!("❌ FAILED: Normalized norm {} doesn't match expected {} (original: {})", 
                       actual_norm, expected_norm, original_norm);
            }
            
            // Verify direction is preserved
            if original_norm > 1e-6 {
                let scale = actual_norm / original_norm;
                for (i, &original) in values.iter().enumerate() {
                    let expected = original * scale;
                    if (normalized[i] - expected).abs() > 1e-5 {
                        panic!("❌ FAILED: Normalized component {} doesn't match expected {}", 
                               normalized[i], expected);
                    }
                }
            }
        }
        println!("✅ L2 normalization produces mathematically correct results");
    }
    
    // Test 3: Gradient Noise MUST Actually Add Noise
    println!("\nTest 3: Verify Gradient Noise Addition");
    {
        // Create identical gradients
        let size = 1000;  // Large enough for statistical tests
        let original_values = vec![1.0; size];
        
        let mut grad1 = Tensor::from_vec(
            original_values.clone(),
            Shape::from_dims(&[size]),
            device.clone()
        )?;
        
        let mut grad2 = Tensor::from_vec(
            original_values.clone(),
            Shape::from_dims(&[size]),
            device.clone()
        )?;
        
        // Add noise to both with same scale
        let noise_scale = 0.1;
        gpu_ops.add_gradient_noise(&mut grad1, noise_scale)?;
        gpu_ops.add_gradient_noise(&mut grad2, noise_scale)?;
        
        let noisy1 = grad1.to_vec()?;
        let noisy2 = grad2.to_vec()?;
        
        // Verify noise was actually added
        let mut unchanged_count = 0;
        for (i, &val) in noisy1.iter().enumerate() {
            if (val - 1.0).abs() < 1e-8 {
                unchanged_count += 1;
            }
        }
        
        if unchanged_count > size / 10 {  // More than 10% unchanged is suspicious
            panic!("❌ FAILED: {} out of {} values unchanged after noise addition!", 
                   unchanged_count, size);
        }
        
        // Verify different noise was added to different tensors
        let mut identical_count = 0;
        for (i, (&v1, &v2)) in noisy1.iter().zip(noisy2.iter()).enumerate() {
            if (v1 - v2).abs() < 1e-8 {
                identical_count += 1;
            }
        }
        
        if identical_count > size / 10 {  // More than 10% identical is suspicious
            panic!("❌ FAILED: Two independent noise additions produced {} identical values out of {}!", 
                   identical_count, size);
        }
        
        // Verify noise has reasonable statistics
        let mean: f32 = noisy1.iter().sum::<f32>() / size as f32;
        let variance: f32 = noisy1.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / size as f32;
        let std_dev = variance.sqrt();
        
        // Mean should still be close to 1.0
        if (mean - 1.0).abs() > 0.05 {
            panic!("❌ FAILED: Mean {} deviates too much from original 1.0", mean);
        }
        
        // Standard deviation should be approximately noise_scale
        if (std_dev - noise_scale).abs() > noise_scale * 0.5 {
            panic!("❌ FAILED: Standard deviation {} doesn't match noise scale {}", 
                   std_dev, noise_scale);
        }
        
        println!("✅ Gradient noise addition works correctly");
        println!("   - Mean: {:.4} (expected ~1.0)", mean);
        println!("   - Std Dev: {:.4} (expected ~{:.4})", std_dev, noise_scale);
    }
    
    // Test 4: Combined Operations Order Matters
    println!("\nTest 4: Verify Operation Order Affects Results");
    {
        let original = vec![5.0, -5.0, 3.0, -3.0];
        
        // Test clip then normalize
        let mut grad1 = Tensor::from_vec(
            original.clone(),
            Shape::from_dims(&[4]),
            device.clone()
        )?;
        gpu_ops.clip_gradient(&mut grad1, 2.0)?;  // Clip to [-2, 2]
        gpu_ops.normalize_gradient(&mut grad1, 3.0)?;  // Then normalize
        let result1 = grad1.to_vec()?;
        
        // Test normalize then clip
        let mut grad2 = Tensor::from_vec(
            original.clone(),
            Shape::from_dims(&[4]),
            device.clone()
        )?;
        gpu_ops.normalize_gradient(&mut grad2, 3.0)?;  // Normalize first
        gpu_ops.clip_gradient(&mut grad2, 2.0)?;  // Then clip
        let result2 = grad2.to_vec()?;
        
        // Results should be different
        let mut same = true;
        for (i, (&v1, &v2)) in result1.iter().zip(result2.iter()).enumerate() {
            if (v1 - v2).abs() > 1e-6 {
                same = false;
                break;
            }
        }
        
        if same {
            panic!("❌ FAILED: Operation order doesn't affect results - operations may be no-ops!");
        }
        
        println!("✅ Operation order correctly affects results");
    }
    
    // Test 5: Error Cases and Edge Cases
    println!("\nTest 5: Verify Error Handling and Edge Cases");
    {
        // Test zero gradient handling
        let mut zero_grad = Tensor::zeros(Shape::from_dims(&[10]), device.clone())?;
        
        // Operations should not crash on zero gradients
        gpu_ops.clip_gradient(&mut zero_grad, 1.0)?;
        gpu_ops.normalize_gradient(&mut zero_grad, 1.0)?;
        
        let zero_result = zero_grad.to_vec()?;
        for &val in &zero_result {
            if val != 0.0 {
                panic!("❌ FAILED: Zero gradient modified to non-zero value {}", val);
            }
        }
        
        // Test single value gradient
        let mut single = Tensor::from_vec(vec![100.0], Shape::from_dims(&[1]), device.clone())?;
        gpu_ops.normalize_gradient(&mut single, 10.0)?;
        let single_result = single.to_vec()?;
        if single_result[0] != 10.0 {
            panic!("❌ FAILED: Single value normalization incorrect: {} (expected 10.0)", 
                   single_result[0]);
        }
        
        println!("✅ Edge cases handled correctly");
    }
    
    // Test 6: GPU Operations Match CPU Implementation
    println!("\nTest 6: Verify GPU Operations Match Expected CPU Results");
    {
        let test_grad = vec![2.5, -3.5, 1.5, -0.5, 4.5];
        let mut gpu_grad = Tensor::from_vec(
            test_grad.clone(),
            Shape::from_dims(&[5]),
            device.clone()
        )?;
        
        // Apply GPU clipping
        gpu_ops.clip_gradient(&mut gpu_grad, 2.0)?;
        let gpu_result = gpu_grad.to_vec()?;
        
        // Compute expected CPU result
        let cpu_result: Vec<f32> = test_grad.iter()
            .map(|&x| x.clamp(-2.0, 2.0))
            .collect();
        
        // Compare
        for (i, (&gpu_val, &cpu_val)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            if (gpu_val - cpu_val).abs() > 1e-6 {
                panic!("❌ FAILED: GPU result {} doesn't match CPU result {} at index {}", 
                       gpu_val, cpu_val, i);
            }
        }
        
        println!("✅ GPU operations match expected CPU computations");
    }
    
    println!("\n🎯 ALL VERIFICATION TESTS PASSED!");
    println!("These tests verify mathematical correctness, not just execution without errors.");
    println!("If any of these tests fail, gradient operations are NOT working correctly.");
    
    Ok(())
}