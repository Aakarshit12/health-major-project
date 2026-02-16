#!/usr/bin/env python
"""Wrapper to run eval.py and capture output"""
import subprocess
import sys

try:
    result = subprocess.run(
        [sys.executable, "eval.py", "--image_dir", "images/"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
    # Also write to file
    with open("eval_run_output.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
        f.write(f"\n\nReturn code: {result.returncode}\n")
    
except Exception as e:
    print(f"Error: {e}")
    with open("eval_run_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}\n")


