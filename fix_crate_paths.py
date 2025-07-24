#!/usr/bin/env python3
"""
Fix incorrect crate paths from the previous script
"""

import os
import re

# Fix patterns
fixes = [
    # Fix incorrect self.crate:: or tensor.crate:: or input.crate:: etc
    (r'(\w+)\.crate::tensor::', r'crate::tensor::'),
    # Fix incorrect alloc_and_copy_to_pool calls with missing Arc
    (r'alloc_and_copy_to_pool\(&device,', r'alloc_and_copy_to_pool(&device,'),
    (r'alloc_and_copy_to_pool\(&self\.device,', r'alloc_and_copy_to_pool(&self.device,'),
]

def fix_file(filepath):
    """Fix crate paths in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
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