#!/usr/bin/env python3
"""
Final fixes for htod_copy_into API
"""

import os
import re

def fix_file(filepath):
    """Fix htod_copy_into patterns"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix alloc_from_pool patterns where it needs mut
    content = re.sub(
        r'let cuda_data = (.*alloc_from_pool.*\));',
        r'let mut cuda_data = \1;',
        content
    )
    
    # Fix .data()_mut() patterns
    content = re.sub(
        r'\.data\(\)_mut\(\)',
        r'.data()?',
        content
    )
    
    # Fix patterns where we need to return Ok after htod_copy_into
    content = re.sub(
        r'device\.htod_copy_into\(([^,]+), &mut cuda_data\)\s*\.map_err\(\|_\| FlameError::CudaDriver\)\?;\s*Ok\(cuda_data\)',
        r'device.htod_copy_into(\1, &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;\n    Ok(cuda_data)',
        content
    )
    
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