#!/usr/bin/env python3
"""
Fix incorrect method calls from tensor.alloc_and_copy_to_pool back to proper allocations
"""

import os
import re

def fix_file(filepath):
    """Fix method calls in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Replace tensor.alloc_and_copy_to_pool(&device, ...) with proper allocation
    # The pattern is: <variable>.alloc_and_copy_to_pool(&device, &<data>)
    content = re.sub(
        r'(\w+)\.alloc_and_copy_to_pool\(&device, &([^)]+)\)',
        r'alloc_from_pool_and_copy(&\1.device, &\2)',
        content
    )
    
    # Also fix self.alloc_and_copy_to_pool patterns
    content = re.sub(
        r'self\.alloc_and_copy_to_pool\(&device, &([^)]+)\)',
        r'alloc_from_pool_and_copy(&self.device, &\1)',
        content
    )
    
    # Fix input.alloc_and_copy_to_pool patterns
    content = re.sub(
        r'input\.alloc_and_copy_to_pool\(&device, &([^)]+)\)',
        r'alloc_from_pool_and_copy(&input.device, &\1)',
        content
    )
    
    # Add helper function if we added calls to it
    if 'alloc_from_pool_and_copy' in content and 'fn alloc_from_pool_and_copy' not in content:
        # Find a good place to insert it (after use statements)
        use_end = content.rfind('\nuse ')
        if use_end != -1:
            line_end = content.find('\n', use_end + 1)
            if line_end != -1:
                helper = '''

// Helper to allocate from pool and copy data
fn alloc_from_pool_and_copy(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<f32>> {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let cuda_data = crate::tensor::alloc_from_pool(device, f32_data.len())?;
    device.htod_copy(&f32_data, &cuda_data)
        .map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}
'''
                content = content[:line_end] + helper + content[line_end:]
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all FLAME source files"""
    src_dir = "flame-core/src"
    
    fixed_files = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                if fix_file(filepath):
                    fixed_files.append(filepath)
    
    print(f"Fixed {len(fixed_files)} files:")
    for f in fixed_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()