#!/bin/bash

# Fix Conv2d::new calls that incorrectly pass bias parameter
echo "Fixing Conv2d::new calls..."

# Find all Rust files
find . -name "*.rs" -type f | while read -r file; do
    # Skip backup files
    if [[ "$file" == *.bak ]]; then
        continue
    fi
    
    # Create backup only if file contains the pattern
    if grep -q "Conv2d::new.*true.*device" "$file" || grep -q "Conv2d::new.*false.*device" "$file"; then
        echo "Fixing Conv2d calls in: $file"
        cp "$file" "$file.conv2d.bak"
        
        # Replace Conv2d::new calls that have the bias parameter
        # Pattern: Conv2d::new(in, out, kernel, stride, padding, true/false, device)
        # Should be: Conv2d::new_with_bias(in, out, kernel, stride, padding, device, true/false)
        sed -i 's/Conv2d::new(\([^,]*\),\([^,]*\),\([^,]*\),\([^,]*\),\([^,]*\), true, \([^)]*\))/Conv2d::new_with_bias(\1,\2,\3,\4,\5, \6, true)/g' "$file"
        sed -i 's/Conv2d::new(\([^,]*\),\([^,]*\),\([^,]*\),\([^,]*\),\([^,]*\), false, \([^)]*\))/Conv2d::new_with_bias(\1,\2,\3,\4,\5, \6, false)/g' "$file"
    fi
done

# Also fix Linear::new calls where device is passed as Arc<Arc<CudaDevice>>
echo "Fixing Linear::new device parameter..."
find . -name "*.rs" -type f | while read -r file; do
    # Skip backup files
    if [[ "$file" == *.bak ]]; then
        continue
    fi
    
    # Look for Linear::new calls with device.clone()
    if grep -q "Linear::new.*device\.clone()" "$file"; then
        echo "Fixing Linear device parameter in: $file"
        cp "$file" "$file.linear.bak"
        
        # Change device.clone() to &device in Linear::new calls
        sed -i 's/Linear::new(\([^,]*\),\([^,]*\),\([^,]*\), device\.clone())/Linear::new(\1,\2,\3, \&device)/g' "$file"
    fi
done

# Fix Tensor::full calls that pass array instead of Shape
echo "Fixing Tensor::full calls..."
find . -name "*.rs" -type f | while read -r file; do
    if [[ "$file" == *.bak ]]; then
        continue
    fi
    
    if grep -q "Tensor::full(&\[" "$file"; then
        echo "Fixing Tensor::full in: $file"
        cp "$file" "$file.full.bak"
        
        # Replace Tensor::full(&[...], ...) with Tensor::full(Shape::from_dims(&[...]), ...)
        sed -i 's/Tensor::full(&\[\([^]]*\)\]/Tensor::full(Shape::from_dims(\&[\1])/g' "$file"
    fi
done

echo "Done! Check the changes and remove backup files if everything looks good."
echo "To remove all backup files: find . -name '*.bak' -delete"