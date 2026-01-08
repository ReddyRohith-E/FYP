#!/usr/bin/env python
"""
ABIDE S3 Setup - Complete Installation Verification

Run this script to verify your S3 setup is working correctly.
"""

import sys
import subprocess
from datetime import datetime

def check_python():
    """Check Python version"""
    version = sys.version.split()[0]
    return f"Python {version}", version >= "3.7"

def check_library(name, description=""):
    """Check if a library is installed"""
    try:
        __import__(name)
        return f"✓ {name}", True
    except ImportError:
        return f"✗ {name} NOT FOUND", False

def check_file(filepath):
    """Check if a file exists"""
    import os
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return f"✓ {filepath} ({size:,} bytes)", True
    else:
        return f"✗ {filepath} MISSING", False

def main():
    print("\n" + "=" * 70)
    print("ABIDE S3 STREAMING SETUP - VERIFICATION REPORT")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    
    # Check Python
    print("1. Python Environment:")
    msg, status = check_python()
    print(f"   {msg}")
    results.append(status)
    
    # Check Libraries
    print("\n2. Required Libraries:")
    libraries = [
        ('boto3', 'AWS S3 SDK'),
        ('botocore', 'AWS core library'),
        ('nibabel', 'NIfTI file format'),
        ('pandas', 'Data analysis'),
        ('numpy', 'Numerical computing'),
        ('fsspec', 'Filesystem abstraction'),
        ('s3fs', 'S3 filesystem'),
    ]
    
    all_libs_ok = True
    for lib, desc in libraries:
        msg, status = check_library(lib, desc)
        print(f"   {msg:30} - {desc}")
        results.append(status)
        all_libs_ok = all_libs_ok and status
    
    # Check Core Files
    print("\n3. Core Python Modules:")
    files = [
        'abide_s3_utils.py',
        'example_s3_usage.py',
        'abide_streaming_analysis.py',
    ]
    
    all_files_ok = True
    for f in files:
        msg, status = check_file(f)
        print(f"   {msg}")
        results.append(status)
        all_files_ok = all_files_ok and status
    
    # Check Documentation
    print("\n4. Documentation Files:")
    docs = [
        'README_S3_SETUP.md',
        'S3_SETUP_GUIDE.md',
        'COMPARISON_LOCAL_VS_STREAMING.md',
        'INDEX.md',
    ]
    
    all_docs_ok = True
    for f in docs:
        msg, status = check_file(f)
        print(f"   {msg}")
        results.append(status)
        all_docs_ok = all_docs_ok and status
    
    # Test S3 Connection
    print("\n5. S3 Connectivity Test:")
    try:
        import boto3
        s3 = boto3.client('s3', region_name='us-east-1')
        response = s3.head_bucket(Bucket='fcp-indi')
        print("   ✓ Connected to ABIDE S3 bucket")
        results.append(True)
    except Exception as e:
        print(f"   ✗ S3 connection failed: {str(e)[:50]}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    
    if passed == total:
        print(f"✓ SETUP COMPLETE! ({passed}/{total} checks passed)")
        print("\nYou are ready to:")
        print("  1. Stream ABIDE fMRI data from S3")
        print("  2. Filter subjects by criteria")
        print("  3. Analyze 900+ subjects without local download")
        print("  4. Process data efficiently in memory")
        print("\nQuick start:")
        print("  python -c \"from abide_s3_utils import quick_load_sample\"")
        print("  samples = quick_load_sample(num_subjects=5)")
        status_code = 0
    else:
        print(f"⚠ SETUP INCOMPLETE ({passed}/{total} checks passed - {percentage:.0f}%)")
        print("\nMissing components:")
        for i, (check, result) in enumerate(zip(
            ["Python", "boto3", "botocore", "nibabel", "pandas", "numpy", 
             "fsspec", "s3fs", "abide_s3_utils.py", "example_s3_usage.py",
             "abide_streaming_analysis.py", "README_S3_SETUP.md", 
             "S3_SETUP_GUIDE.md", "COMPARISON_LOCAL_VS_STREAMING.md",
             "INDEX.md", "S3 Connection"],
            results
        ), 1):
            if not result:
                print(f"  - {check}")
        
        print("\nNext steps:")
        print("  pip install boto3 nibabel pandas numpy fsspec s3fs")
        print("  Create missing files (see INDEX.md)")
        status_code = 1
    
    print("=" * 70 + "\n")
    return status_code

if __name__ == '__main__':
    sys.exit(main())
