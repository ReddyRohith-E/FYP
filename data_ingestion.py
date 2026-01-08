import os
import boto3
import urllib.request
from pathlib import Path
import pandas as pd
from nilearn import datasets
import aws_config

def download_abide_subset(n_subjects=10, data_dir='./data'):
    """
    Downloads a small subset of ABIDE I data for testing the pipeline.
    Uses nilearn to fetch data.
    """
    print(f"Downloading {n_subjects} subjects from ABIDE I...")
    # Fetch ABIDE data
    # We choose 'rois_ho' for region of interest data if we were doing simple ML, 
    # but for 3D CNN we usually want raw or preprocessed NIfTI.
    # Nilearn fetch_abide_pcp returns file paths to NIfTI images.
    
    # fetching raw functional MRI data (preprocessed)
    abide_data = datasets.fetch_abide_pcp(
        n_subjects=n_subjects, 
        pipeline='cpac', 
        band_pass_filtering=True, 
        global_signal_regression=False, 
        quality_checked=True,
        data_dir=data_dir
    )
    
    return abide_data

def upload_to_s3(local_path, bucket, s3_key):
    """Uploads a file to S3."""
    s3_client = boto3.client('s3', region_name=aws_config.REGION)
    try:
        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")

def prepare_data_and_upload():
    """Main function to prepare data and upload to S3."""
    
    # 1. Download Data locally
    data = download_abide_subset()
    func_files = data.func_preproc
    pheno_file = data.phenotypic
    
    # 2. Upload to S3
    bucket_name = aws_config.BUCKET_NAME
    prefix = aws_config.PREFIX
    
    # Ensure bucket exists (or create it)
    s3_resource = boto3.resource('s3', region_name=aws_config.REGION)
    bucket = s3_resource.Bucket(bucket_name)
    
    if not bucket.creation_date:
        try:
            if aws_config.REGION == 'us-east-1':
                s3_resource.create_bucket(Bucket=bucket_name)
            else:
                s3_resource.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': aws_config.REGION}
                )
            print(f"Created bucket: {bucket_name}")
        except Exception as e:
            print(f"Bucket might already exist or error creating: {e}")

    # Upload NIfTI files
    s3_input_uris = []
    
    for local_file in func_files:
        filename = os.path.basename(local_file)
        # Use subject ID in path if possible, but filename usually contains it
        s3_key = f"{prefix}/raw/{filename}"
        upload_to_s3(local_file, bucket_name, s3_key)
        s3_input_uris.append(f"s3://{bucket_name}/{s3_key}")

    print("Data upload complete.")
    return s3_input_uris

if __name__ == "__main__":
    prepare_data_and_upload()
