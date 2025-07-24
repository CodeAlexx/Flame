#!/usr/bin/env python3
"""
Script to update direct allocations to use memory pool
"""

import os
import re

replacements = [
    # device.alloc_zeros
    (r'device\.alloc_zeros::<f32>\(([^)]+)\)', r'crate::tensor::alloc_zeros_from_pool(&device, \1)'),
    (r'self\.device\.alloc_zeros::<f32>\(([^)]+)\)', r'crate::tensor::alloc_zeros_from_pool(&self.device, \1)'),
    (r'input\.device\.alloc_zeros::<f32>\(([^)]+)\)', r'crate::tensor::alloc_zeros_from_pool(&input.device, \1)'),
    (r'a\.device\.alloc_zeros::<f32>\(([^)]+)\)', r'crate::tensor::alloc_zeros_from_pool(&a.device, \1)'),
    
    # unsafe device.alloc
    (r'unsafe\s*\{\s*device\.alloc::<f32>\(([^)]+)\)\s*\}', r'crate::tensor::alloc_from_pool(&device, \1)'),
    (r'unsafe\s*\{\s*self\.device\.alloc::<f32>\(([^)]+)\)\s*\}', r'crate::tensor::alloc_from_pool(&self.device, \1)'),
    (r'unsafe\s*\{\s*input\.device\.alloc::<f32>\(([^)]+)\)\s*\}', r'crate::tensor::alloc_from_pool(&input.device, \1)'),
    (r'unsafe\s*\{\s*a\.device\.alloc::<f32>\(([^)]+)\)\s*\}', r'crate::tensor::alloc_from_pool(&a.device, \1)'),
    
    # htod_sync_copy for Vec<f32>
    (r'device\.htod_sync_copy\(&([^)]+)\)', r'alloc_and_copy_to_pool(&device, &\1)'),
    (r'self\.device\.htod_sync_copy\(&([^)]+)\)', r'alloc_and_copy_to_pool(&self.device, &\1)'),
]

# Files to skip
skip_files = [
    'tensor_original.rs',  # Old implementation
    'cuda_tensor_gpu.rs',  # Old implementation
    'memory_pool.rs',      # The pool itself
    'tensor.rs',           # Already updated
]

def update_file(filepath):
    """Update allocations in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Add helper function at top of file if needed
    if 'alloc_and_copy_to_pool' in content and 'fn alloc_and_copy_to_pool' not in content:
        # Find where to insert - after use statements
        use_end = content.rfind('\nuse ')
        if use_end != -1:
            # Find the end of that line
            line_end = content.find('\n', use_end + 1)
            if line_end != -1:
                helper = '''

// Helper function for allocating and copying to GPU via memory pool
fn alloc_and_copy_to_pool<T: AsRef<[f32]>>(device: &Arc<CudaDevice>, data: T) -> Result<CudaSlice<f32>> {
    let slice = data.as_ref();
    let cuda_data = crate::tensor::alloc_from_pool(device, slice.len())?;
    device.htod_copy(slice, &cuda_data)
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
    """Update all FLAME source files"""
    src_dir = "flame-core/src"
    
    updated_files = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.rs') and file not in skip_files:
                filepath = os.path.join(root, file)
                if update_file(filepath):
                    updated_files.append(filepath)
    
    print(f"Updated {len(updated_files)} files:")
    for f in updated_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()