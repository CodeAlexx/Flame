use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::activations::{LeakyReLU, ELU, PReLU, ReLU, GELU, SiLU, Tanh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame activation functions...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Create test input with positive and negative values
    let input_data = vec![
        -2.0, -1.5, -1.0, -0.5,
         0.0,  0.5,  1.0,  1.5,
         2.0,  2.5,  3.0, -3.0,
    ];
    
    let input = Tensor::from_vec(
        input_data.clone(),
        Shape::from_dims(&[1, 3, 2, 2]),
        device.clone()
    )?;
    println!("Input shape: {:?}", input.shape().dims());
    println!("Input values: {:?}", input_data);
    
    // Test ReLU
    println!("\n--- Testing ReLU ---");
    let relu = ReLU::new();
    let relu_output = relu.forward(&input)?;
    let relu_vec = relu_output.to_vec()?;
    println!("ReLU output: {:?}", relu_vec);
    println!("Expected: negative values become 0");
    
    // Test LeakyReLU
    println!("\n--- Testing LeakyReLU (slope=0.1) ---");
    let leaky_relu = LeakyReLU::new(0.1);
    let leaky_output = leaky_relu.forward(&input)?;
    let leaky_vec = leaky_output.to_vec()?;
    println!("LeakyReLU output: {:?}", leaky_vec);
    println!("Expected: negative values multiplied by 0.1");
    
    // Test ELU
    println!("\n--- Testing ELU (alpha=1.0) ---");
    let elu = ELU::new(1.0);
    let elu_output = elu.forward(&input)?;
    let elu_vec = elu_output.to_vec()?;
    println!("ELU output: {:?}", elu_vec);
    println!("Expected: negative values become alpha*(exp(x)-1)");
    
    // Test PReLU
    println!("\n--- Testing PReLU ---");
    let prelu = PReLU::new(3, device.clone())?;
    let prelu_output = prelu.forward(&input)?;
    let prelu_vec = prelu_output.to_vec()?;
    println!("PReLU output: {:?}", prelu_vec);
    println!("PReLU weights: {:?}", prelu.weight.to_vec()?);
    println!("Expected: negative values multiplied by learnable weight (0.25)");
    
    // Test GELU
    println!("\n--- Testing GELU ---");
    let gelu = GELU::new();
    let gelu_output = gelu.forward(&input)?;
    let gelu_vec = gelu_output.to_vec()?;
    println!("GELU output: {:?}", gelu_vec);
    println!("Expected: smooth approximation of ReLU");
    
    // Test SiLU/Swish
    println!("\n--- Testing SiLU (Swish) ---");
    let silu = SiLU::new();
    let silu_output = silu.forward(&input)?;
    let silu_vec = silu_output.to_vec()?;
    println!("SiLU output: {:?}", silu_vec);
    println!("Expected: x * sigmoid(x)");
    
    // Test Tanh
    println!("\n--- Testing Tanh ---");
    let tanh = Tanh::new();
    let tanh_output = tanh.forward(&input)?;
    let tanh_vec = tanh_output.to_vec()?;
    println!("Tanh output: {:?}", tanh_vec);
    println!("Expected: values in range (-1, 1)");
    
    // Test with different input shapes
    println!("\n--- Testing with larger input ---");
    let larger_input = Tensor::randn(
        Shape::from_dims(&[2, 16, 8, 8]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let relu_large = relu.forward(&larger_input)?;
    println!("Large input shape: {:?}", larger_input.shape().dims());
    println!("ReLU output shape: {:?}", relu_large.shape().dims());
    
    // Verify PReLU with correct number of channels
    let prelu_16 = PReLU::new(16, device)?;
    let prelu_large = prelu_16.forward(&larger_input)?;
    println!("PReLU output shape: {:?}", prelu_large.shape().dims());
    
    println!("\nAll activation tests completed!");
    
    Ok(())
}