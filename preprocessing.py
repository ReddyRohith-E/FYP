
import argparse
import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def load_and_preprocess_image(filepath, target_shape=(64, 64, 64)):
    """
    Loads a NIfTI file, resizes it to target_shape, and normalizes intensity.
    """
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Handle 4D fMRI data (Time series)
        # For simplicity in this demo, we might take the mean over time or just the first frame
        # Or if it's 3D already (structural), we keep it.
        # ABIDE func_preproc is 4D (x, y, z, time).
        
        if len(data.shape) == 4:
            # Strategy: Calculate Mean Image over time to get a 3D volume
            # Alternatively: Use a CRNN or 3D CNN on chunks. 
            # For this MVP 3D CNN, we'll use the Temporal Mean.
            data = np.mean(data, axis=-1)
            
        # Resize/Resample
        # Calculate zoom factors
        zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
        data_resized = zoom(data, zoom_factors, order=1) # Linear interpolation
        
        # Normalize (Z-score or MinMax)
        # robust scaling
        p2, p98 = np.percentile(data_resized, (2, 98))
        data_resized = np.clip(data_resized, p2, p98)
        
        if p98 - p2 > 0:
            data_resized = (data_resized - p2) / (p98 - p2)
        else:
            data_resized = np.zeros_like(data_resized)

        return data_resized
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/opt/ml/processing')
    args = parser.parse_args()
    
    input_dir = f"{args.base_dir}/input"
    output_dir = f"{args.base_dir}/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate all NIfTI files
    file_paths = glob.glob(os.path.join(input_dir, "**/*.nii.gz"), recursive=True)
    
    print(f"Found {len(file_paths)} files to process.")
    
    metadata = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        subject_id = filename.split('_')[0] # Heuristic extract ID
        
        processed_data = load_and_preprocess_image(file_path)
        
        if processed_data is not None:
            # Save as numpy array
            output_filename = filename.replace('.nii.gz', '.npy')
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, processed_data)
            print(f"Saved {output_path}")
            
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
