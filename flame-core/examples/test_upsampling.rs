use flame_core::{Tensor, Shape, CudaDevice};
use flame_core::upsampling::{
    Upsample2d, Upsample2dConfig, UpsampleMode,
    ConvTranspose2d, ConvTranspose2dConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Flame upsampling layers...");
    
    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("CUDA device initialized");
    
    // Create test input tensor [batch=1, channels=1, height=2, width=2]
    let input_data = vec![
        1.0, 2.0,
        3.0, 4.0,
    ];
    
    let input = Tensor::from_vec(
        input_data,
        Shape::from_dims(&[1, 1, 2, 2]),
        device.clone()
    )?;
    println!("Input shape: {:?}", input.shape().dims());
    
    // Test nearest neighbor upsampling
    println!("\n--- Testing Nearest Upsampling ---");
    let nearest_config = Upsample2dConfig::new(UpsampleMode::Nearest)
        .with_size((4, 4));
    let nearest_upsample = Upsample2d::new(nearest_config);
    let nearest_output = nearest_upsample.forward(&input)?;
    println!("Nearest output shape: {:?}", nearest_output.shape().dims());
    
    let nearest_vec = nearest_output.to_vec()?;
    println!("Nearest output values:");
    for i in 0..4 {
        println!("  {:?}", &nearest_vec[i*4..(i+1)*4]);
    }
    println!("Expected: each 2x2 block should repeat the input value");
    
    // Test bilinear upsampling
    println!("\n--- Testing Bilinear Upsampling ---");
    let bilinear_config = Upsample2dConfig::new(UpsampleMode::Bilinear)
        .with_scale_factor((2.0, 2.0));
    let bilinear_upsample = Upsample2d::new(bilinear_config);
    let bilinear_output = bilinear_upsample.forward(&input)?;
    println!("Bilinear output shape: {:?}", bilinear_output.shape().dims());
    
    let bilinear_vec = bilinear_output.to_vec()?;
    println!("Bilinear output values:");
    for i in 0..4 {
        println!("  {:?}", &bilinear_vec[i*4..(i+1)*4]);
    }
    println!("Expected: smoothly interpolated values");
    
    // Test different input size
    println!("\n--- Testing with larger input ---");
    let larger_input = Tensor::randn(
        Shape::from_dims(&[2, 3, 8, 8]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let config = Upsample2dConfig::new(UpsampleMode::Nearest)
        .with_scale_factor((2.0, 2.0));
    let upsample = Upsample2d::new(config);
    let output = upsample.forward(&larger_input)?;
    println!("Large input shape: {:?}", larger_input.shape().dims());
    println!("Large output shape: {:?}", output.shape().dims());
    
    // Test ConvTranspose2d
    println!("\n--- Testing ConvTranspose2d ---");
    let conv_input = Tensor::randn(
        Shape::from_dims(&[1, 16, 4, 4]),
        0.0,
        1.0,
        device.clone()
    )?;
    
    let conv_config = ConvTranspose2dConfig::new(16, 32, (3, 3));
    let conv_transpose = ConvTranspose2d::new(conv_config, device.clone())?;
    let conv_output = conv_transpose.forward(&conv_input)?;
    println!("ConvTranspose2d input shape: {:?}", conv_input.shape().dims());
    println!("ConvTranspose2d output shape: {:?}", conv_output.shape().dims());
    println!("Expected output shape: [1, 32, 6, 6]");
    
    // Test with stride
    println!("\n--- Testing ConvTranspose2d with stride ---");
    let mut conv_config2 = ConvTranspose2dConfig::new(16, 32, (3, 3));
    conv_config2.stride = (2, 2);
    let conv_transpose2 = ConvTranspose2d::new(conv_config2, device)?;
    let conv_output2 = conv_transpose2.forward(&conv_input)?;
    println!("ConvTranspose2d with stride=2 output shape: {:?}", conv_output2.shape().dims());
    println!("Expected output shape: [1, 32, 9, 9]");
    
    println!("\nAll upsampling tests completed!");
    
    Ok(())
}