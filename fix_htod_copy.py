#!/usr/bin/env python3
"""
Fix htod_copy API changes - the new API puts the data into the slice directly
"""

import os
import re

def fix_file(filepath):
    """Fix htod_copy calls in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix htod_copy patterns
    # Pattern: device.htod_copy(&data, &cuda_data) -> device.htod_copy_into(&data, &mut cuda_data) 
    content = re.sub(
        r'device\.htod_copy\(([^,]+),\s*&cuda_data\)',
        r'device.htod_copy_into(\1, &mut cuda_data)',
        content
    )
    
    # Fix self.device patterns
    content = re.sub(
        r'self\.device\.htod_copy\(([^,]+),\s*&([^)]+)\)',
        r'self.device.htod_copy_into(\1, &mut \2)',
        content
    )
    
    # Fix tensor.device patterns
    content = re.sub(
        r'(\w+)\.device\.htod_copy\(([^,]+),\s*&([^)]+)\)',
        r'\1.device.htod_copy_into(\2, &mut \3)',
        content
    )
    
    # Fix TensorStorage as_slice calls that return references
    content = re.sub(
        r'(\w+)\.storage\.as_slice\(\)\?',
        r'\1.storage.as_slice()',
        content
    )
    
    # Fix data field access to method calls
    content = re.sub(
        r'\.data(?!\()',
        r'.data()',
        content
    )
    
    # Fix data_mut() calls to data()
    content = re.sub(
        r'\.data_mut\(\)',
        r'.data()',
        content
    )
    
    # Fix device_ptr() to device_ptr_mut()
    content = re.sub(
        r'\.device_ptr\(\)',
        r'.device_ptr_mut()',
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