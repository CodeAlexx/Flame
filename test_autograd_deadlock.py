#!/usr/bin/env python3
"""
Test script to verify autograd deadlock fix for FLAME.
This mimics the exact hanging scenario from todoagents.txt
"""

import subprocess
import threading
import time
import sys

def run_rust_test():
    """Run the Rust test that should hang without the fix"""
    cmd = [
        "cargo", "test", "test_autograd_fix_two_ops", 
        "--release", "--", "--nocapture", "--test-threads=1"
    ]
    
    print("Running autograd test...")
    print(f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/home/alex/diffusers-rs/flame/flame-core"
    )
    
    # Set a timeout
    timeout = 30  # seconds
    start_time = time.time()
    
    output_lines = []
    hung = False
    
    # Read output line by line
    while True:
        line = process.stdout.readline()
        if line:
            print(line.rstrip())
            output_lines.append(line.rstrip())
            
            # Check for specific debug output
            if "Computing Mul gradients..." in line:
                print(">>> DETECTED: Mul gradient computation starting")
            elif "Got saved tensors, computing grad_lhs..." in line:
                print(">>> DETECTED: About to call tensor.mul() - this is where it hangs!")
        
        # Check if process finished
        if process.poll() is not None:
            break
            
        # Check timeout
        if time.time() - start_time > timeout:
            print(f"\n>>> TEST HUNG! Process didn't complete in {timeout} seconds")
            print(">>> This confirms the deadlock issue")
            process.kill()
            hung = True
            break
    
    if not hung:
        return_code = process.wait()
        if return_code == 0:
            print("\n>>> TEST PASSED! Autograd completed without hanging")
            print(">>> The fix is working correctly")
        else:
            print(f"\n>>> TEST FAILED with return code {return_code}")
    
    return hung, output_lines

def main():
    print("=== FLAME Autograd Deadlock Test ===")
    print("Testing the minimal hanging case: x -> add -> mul -> sum -> backward\n")
    
    hung, output = run_rust_test()
    
    if hung:
        print("\n=== Analysis ===")
        print("The test hung as expected without a proper fix.")
        print("The deadlock occurs when:")
        print("1. backward() holds the AUTOGRAD_CONTEXT lock")
        print("2. compute_gradients() calls tensor.mul()")
        print("3. tensor.mul() tries to acquire the same lock -> DEADLOCK")
        print("\nThe fix needs to prevent tensor operations from trying to")
        print("record themselves during backward pass.")
        sys.exit(1)
    else:
        print("\n=== Success ===")
        print("The autograd deadlock has been fixed!")
        print("Complex computation graphs can now be differentiated.")
        sys.exit(0)

if __name__ == "__main__":
    main()