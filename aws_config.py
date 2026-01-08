import os
import sys
import argparse
import boto3
from boto3.session import Session as Boto3Session
try:
    # Prefer direct import to avoid missing attribute on package namespace
    from sagemaker import get_execution_role as _get_execution_role
except Exception:
    _get_execution_role = None

# Load .env if available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Configuration ---
REGION = 'us-east-1' # Change to your preferred region
BUCKET_NAME = 'asd-abide-data-pipeline' # Replace with a unique bucket name
PREFIX = 'abide-project'

# IAM Role for SageMaker:
# - In SageMaker-managed environments, this is auto-detected via get_execution_role()
# - Locally, fall back to env var SAGEMAKER_EXECUTION_ROLE_ARN or leave as None
ROLE = None
if _get_execution_role is not None:
    try:
        ROLE = _get_execution_role()
    except Exception:
        ROLE = None

if not ROLE:
    ROLE = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
    if not ROLE:
        print(
            "SageMaker execution role not detected. Set SAGEMAKER_EXECUTION_ROLE_ARN "
            "or update ROLE in aws_config.py when creating SageMaker resources."
        )


def _extract_role_name_from_arn(role_arn: str) -> str | None:
    if not role_arn or ":role/" not in role_arn:
        return None
    # Support possible path prefixes in role name
    return "/".join(role_arn.split(":role/")[-1].split("/"))


def _session(profile: str | None) -> Boto3Session:
    return Boto3Session(profile_name=profile) if profile else Boto3Session()


def verify_setup(region: str = REGION, role_arn: str | None = ROLE, profile: str | None = None) -> int:
    print("Verifying AWS credentials and SageMaker access...")
    session = _session(profile)
    # 1) STS identity
    try:
        sts = session.client("sts", region_name=region)
        ident = sts.get_caller_identity()
        print(f"- STS identity OK: Account={ident.get('Account')} UserId={ident.get('UserId')}")
    except Exception as e:
        print("- STS identity FAILED:", e)
        print("  Ensure AWS credentials are configured (env vars, shared config, or SSO).")
        return 2

    # 2) IAM role existence (if provided)
    if role_arn:
        try:
            iam = session.client("iam")
            role_name = _extract_role_name_from_arn(role_arn)
            if not role_name:
                raise ValueError("Could not parse role name from ARN")
            resp = iam.get_role(RoleName=role_name)
            print(f"- IAM role OK: {resp['Role']['Arn']}")
        except Exception as e:
            print("- IAM role check FAILED:", e)
            print("  Verify SAGEMAKER_EXECUTION_ROLE_ARN and IAM permissions to read the role.")
            return 3
    else:
        print("- No ROLE set; skipping IAM role check.")

    # 3) SageMaker basic permission check
    try:
        sm = session.client("sagemaker", region_name=region)
        sm.list_training_jobs(MaxResults=1)
        print("- SageMaker permissions OK: able to list training jobs.")
    except Exception as e:
        print("- SageMaker permission check FAILED:", e)
        print("  Ensure your credentials (or role) have SageMaker read permissions.")
        return 4

    print("Verification successful.")
    return 0


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="AWS configuration helper for SageMaker")
    parser.add_argument("--verify", action="store_true", help="Verify AWS credentials and SageMaker access")
    parser.add_argument("--profile", type=str, default=None, help="AWS named profile to use for verification")
    args = parser.parse_args(argv)

    print(f"Region: {REGION}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Prefix: {PREFIX}")
    print(f"Role:   {ROLE if ROLE else 'None'}")

    if args.verify:
        return verify_setup(REGION, ROLE, profile=args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

# Instance Types
PROCESSING_INSTANCE_TYPE = 'ml.m5.large'
TRAINING_INSTANCE_TYPE = 'ml.g4dn.xlarge' # GPU instance for training
INFERENCE_INSTANCE_TYPE = 'ml.m5.large'

# Local paths
LOCAL_DATA_DIR = './data'
