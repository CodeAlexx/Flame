#!/bin/bash

# FLAME Test Runner Script
# This script runs all FLAME tests and measures performance

# Set up CUDA environment
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Enable backtrace for better error reporting
export RUST_BACKTRACE=1

# Create test results directory
mkdir -p test_results
timestamp=$(date +%Y%m%d_%H%M%S)
results_dir="test_results/run_$timestamp"
mkdir -p "$results_dir"

# Function to run a test and capture results
run_test() {
    local test_name=$1
    local test_type=$2
    local output_file="$results_dir/${test_name}_output.txt"
    local timing_file="$results_dir/${test_name}_timing.txt"
    
    echo "Running $test_type: $test_name..."
    
    # Measure execution time and memory
    /usr/bin/time -v cargo $test_type $test_name --release 2>&1 | tee "$output_file" > "$timing_file" 2>&1
    
    # Extract key metrics
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $test_name: PASSED"
    else
        echo "✗ $test_name: FAILED (exit code: $exit_code)"
    fi
    
    # Extract timing and memory info
    if grep -q "Elapsed (wall clock) time" "$timing_file"; then
        elapsed=$(grep "Elapsed (wall clock) time" "$timing_file" | cut -d: -f2-)
        max_memory=$(grep "Maximum resident set size" "$timing_file" | cut -d: -f2)
        echo "  Time: $elapsed"
        echo "  Memory: $max_memory KB"
    fi
    
    echo ""
}

# Header
echo "FLAME Test Suite Execution Report"
echo "================================="
echo "Date: $(date)"
echo "CUDA Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo ""

# Run unit tests
echo "Running Unit Tests..."
echo "--------------------"
cargo test --release 2>&1 | tee "$results_dir/unit_tests_output.txt"
echo ""

# Run integration tests in tests/ directory
echo "Running Integration Tests..."
echo "---------------------------"
for test_file in tests/*.rs; do
    test_name=$(basename "$test_file" .rs)
    run_test "$test_name" "test --test"
done

# Run examples
echo "Running Example Tests..."
echo "-----------------------"
# List of key examples to run
examples=(
    "simple_flame_test"
    "tensor_ops"
    "autograd_demo"
    "conv2d_demo"
    "activation_test"
    "optimizer_test"
    "attention_test"
    "gradient_clip_test"
    "mixed_precision_test"
    "test_gradient_simple"
    "test_pooling_correct"
    "test_norm"
    "test_activations"
)

for example in "${examples[@]}"; do
    run_test "$example" "run --example"
done

# Generate summary report
echo "Generating Summary Report..."
summary_file="$results_dir/summary.txt"
{
    echo "FLAME Test Results Summary"
    echo "========================="
    echo "Date: $(date)"
    echo ""
    
    echo "Test Results:"
    echo "-------------"
    passed=$(grep -c "✓" "$results_dir"/*_output.txt 2>/dev/null || echo 0)
    failed=$(grep -c "✗" "$results_dir"/*_output.txt 2>/dev/null || echo 0)
    echo "Passed: $passed"
    echo "Failed: $failed"
    echo ""
    
    echo "Performance Metrics:"
    echo "-------------------"
    # Aggregate timing data
    for timing_file in "$results_dir"/*_timing.txt; do
        if [ -f "$timing_file" ]; then
            test_name=$(basename "$timing_file" _timing.txt)
            if grep -q "Elapsed (wall clock) time" "$timing_file"; then
                elapsed=$(grep "Elapsed (wall clock) time" "$timing_file" | cut -d: -f2-)
                echo "$test_name: $elapsed"
            fi
        fi
    done
} > "$summary_file"

echo ""
echo "Test execution complete!"
echo "Results saved to: $results_dir"
echo "Summary available at: $summary_file"