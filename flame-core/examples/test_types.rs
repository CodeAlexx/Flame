use flame_core::CudaDevice;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test what type CudaDevice::new returns
    let device = CudaDevice::new(0)?;
    println!("Type of device: {}", std::any::type_name_of_val(&device));
    
    let device_arc = Arc::new(device);
    println!("Type of device_arc: {}", std::any::type_name_of_val(&device_arc));
    
    let cloned = device_arc.clone();
    println!("Type of cloned: {}", std::any::type_name_of_val(&cloned));
    
    Ok(())
}