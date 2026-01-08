# ABIDE S3 Setup Guide - Zero Local Storage Download

You now have S3 access configured for the complete ABIDE dataset. Here's what you can do:

## Quick Start

### Option 1: Load samples instantly (no storage needed)

```python
from abide_s3_utils import quick_load_sample

# Load 5 sample subjects in memory only
samples = quick_load_sample(num_subjects=5)
for result in samples:
    subject_id = result['subject_id']
    nifti_data = result['nifti']
    phenotypic = result['phenotypic']
```

### Option 2: Stream specific subjects by criteria

```python
from abide_s3_utils import S3ABIDEClient, ABIDEDataFilter

client = S3ABIDEClient()
pheno_df = client.get_phenotypic_data()

# Filter: Age 5-10, ASD diagnosis, NYU site
filter_obj = ABIDEDataFilter(pheno_df)
filtered = filter_obj.apply_filters(
    age_range=(5, 10),
    diagnosis=1,  # 1=ASD, 2=TDC
    site='NYU',
    max_motion=0.5
)

# Load matching subjects one-by-one (streamed from S3)
results = client.batch_load_subjects(
    filtered['FILE_ID'].tolist(),
    max_subjects=10
)
```

### Option 3: Load and analyze directly

```python
# Get specific subject
phenotypic_data, nifti_img = client.get_subject_data('Pitt_0050004')

# Process without saving
data_array = nifti_img.get_fdata()
print(f"Shape: {data_array.shape}")
print(f"Mean intensity: {data_array.mean()}")
```

## Available Sites in ABIDE

- **Pitt** - University of Pittsburgh Medical Center
- **NYU** - New York University
- **OHSU** - Oregon Health and Science University
- **UCLA** - University of California Los Angeles
- **Stanford** - Stanford University
- **Caltech** - California Institute of Technology
- **Yale** - Yale University
- **KKI** - Kennedy Krieger Institute
- **Leuven** - KU Leuven (Belgium)
- **Olin** - Washington University
- **SDSU** - San Diego State University
- **Trinity** - Trinity College Dublin
- **CMU** - Carnegie Mellon University
- **MaxMun** - Max Planck Institute (Munich)
- **SBL** - University of Quebec
- **UM_1, UM_2** - University of Michigan
- **USM** - University of São Paulo Medical School

## Data Available

- **Pipeline**: CPAC (Configurable Pipeline for the Analysis of Connectomes)
- **Strategy**: `nofilt_noglobal` (no filtering, no global signal regression)
- **Derivative**: `func_preproc` (preprocessed functional MRI)
- **Format**: NIfTI gzip compressed (.nii.gz)
- **Size per file**: ~200-500 MB
- **Total subjects**: 900+
- **Total size if downloaded**: 180GB+

## Phenotypic Data Available

Over 90 variables per subject including:

- Demographics: Age, Sex, Handedness
- Diagnosis: DX_GROUP (1=ASD, 2=TDC)
- Assessments: ADOS, ADI-R, SRS, SCQ scores
- IQ measures: WISC-IV, WASI scores
- Quality metrics: Head motion (mean FD), brain extraction metrics
- Medical history: Medication status, comorbidities

## Storage Cost

### Local Download Approach

- **Storage needed**: 180GB+ on your machine
- **Download time**: 8-24 hours
- **Cost**: Disk space

### S3 Streaming Approach (What we just set up)

- **Storage needed**: ~1GB for analysis (no raw data)
- **Transfer time**: Seconds per subject (on-demand)
- **Cost**: AWS data transfer charges (~$0.02 per GB accessed)
  - Loading 100 subjects = ~30GB transfer = ~$0.60

## Example: Typical Workflow

```python
from abide_s3_utils import S3ABIDEClient, ABIDEDataFilter
import numpy as np

# Initialize
client = S3ABIDEClient()
pheno = client.get_phenotypic_data()
filter_obj = ABIDEDataFilter(pheno)

# Get children (5-15 years) with ASD
children_asd = filter_obj.apply_filters(
    age_range=(5, 15),
    diagnosis=1
)

print(f"Found {len(children_asd)} children with ASD")

# Process first 20
subject_ids = children_asd['FILE_ID'].head(20).tolist()
results = client.batch_load_subjects(subject_ids, max_subjects=20)

# Analyze
features = []
for result in results:
    nifti = result['nifti']
    data = nifti.get_fdata()

    # Calculate features
    features.append({
        'subject_id': result['subject_id'],
        'mean_activation': data.mean(),
        'std_activation': data.std(),
        'shape': data.shape
    })

# No files saved locally - everything streamed!
```

## Performance Tips

1. **Filter first** - Use `ABIDEDataFilter` to reduce number of subjects
2. **Batch load** - Use `batch_load_subjects()` for multiple subjects
3. **Cloud EC2** - If in AWS, use EC2 in `us-east-1` region for best speed
4. **Cache phenotypic** - Phenotypic data is cached after first load
5. **Check motion** - Use `max_motion` filter to get quality subjects only

## Troubleshooting

**Error: No credentials found**

- Use anonymous access (default for ABIDE public S3)
- No AWS account needed for read-only access

**Slow downloads**

- Check internet connection
- ABIDE bucket is in `us-east-1` - closer regions = faster
- Batch multiple subjects together

**Out of memory**

- Reduce number of subjects loaded at once
- Process NIfTI data in chunks using nibabel

## Files Created

- `abide_s3_utils.py` - Main S3 utility module
- `example_s3_usage.py` - Example code with 5 usage patterns
- `S3_SETUP_GUIDE.md` - This file

## Next Steps

1. **Test it**: Run `example_s3_usage.py` to test access
2. **Modify filtering**: Adjust criteria in `ABIDEDataFilter` for your needs
3. **Integrate**: Use in your analysis/ML pipelines
4. **Scale**: Process hundreds of subjects on-demand

## Security Notes

- ABIDE data is **public** - no special permissions needed
- Uses **anonymous S3 access** - no credentials stored
- Data is **read-only** - cannot modify S3 files
- Access logs may record IP addresses (normal for AWS)

## References

- ABIDE Initiative: http://fcon_1000.projects.nitrc.org/indi/abide/
- CPAC Pipeline: https://fcp-indi.github.io/docs/latest/
- Boto3 Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/
- NiBabel Documentation: https://nipy.org/nibabel/

## Support

For issues with:

- **ABIDE data**: See NITRC ABIDE forum
- **boto3/S3**: AWS documentation
- **nibabel**: NiBabel GitHub issues
- **Pipeline**: CPAC documentation
