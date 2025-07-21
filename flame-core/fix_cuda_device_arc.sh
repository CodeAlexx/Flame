#!/bin/bash

# Fix the double Arc wrapping issue for CudaDevice
# In cudarc 0.11.9, CudaDevice::new() already returns Arc<CudaDevice>

echo "Fixing Arc<Arc<CudaDevice>> issue..."

# Replace Arc::new(CudaDevice::new(...)) with just CudaDevice::new(...)
find . -name "*.rs" -type f | while read -r file; do
    # Skip backup files
    if [[ "$file" == *.bak ]]; then
        continue
    fi
    
    # Check if file contains the pattern
    if grep -q "Arc::new(CudaDevice::new" "$file"; then
        echo "Fixing: $file"
        # Create backup
        cp "$file" "$file.bak"
        
        # Replace the pattern - handle both single line and multi-line cases
        sed -i 's/Arc::new(CudaDevice::new(\([^)]*\))\([^)]*\))/CudaDevice::new(\1)\2/g' "$file"
        
        # Also handle cases with .expect or .unwrap after the closing paren
        sed -i 's/Arc::new(CudaDevice::new(\([^)]*\))\.expect(\([^)]*\)))/CudaDevice::new(\1).expect(\2)/g' "$file"
        sed -i 's/Arc::new(CudaDevice::new(\([^)]*\))\.unwrap())/CudaDevice::new(\1).unwrap()/g' "$file"
        
        # Handle cases with ? operator
        sed -i 's/Arc::new(CudaDevice::new(\([^)]*\))?)/CudaDevice::new(\1)?/g' "$file"
    fi
done

echo "Done! Check the changes and remove .bak files if everything looks good."
echo "To remove backup files: find . -name '*.rs.bak' -delete"