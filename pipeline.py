
import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
import aws_config

def get_pipeline(
    region,
    role,
    default_bucket,
    pipeline_name="ASD-ABIDE-Pipeline",
    base_job_prefix="asd-abide"
):
    """
    Gets a SageMaker Pipeline instance.
    """
    
    # -------------------------------------------------------------------------
    # 1. Parameters
    # -------------------------------------------------------------------------
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value=aws_config.PROCESSING_INSTANCE_TYPE
    )
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=aws_config.TRAINING_INSTANCE_TYPE
    )
    input_data_uri = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/{aws_config.PREFIX}/raw" # Default to raw data location
    )
    
    # -------------------------------------------------------------------------
    # 2. Processing Step
    # -------------------------------------------------------------------------
    # We use a generic ScriptProcessor or SKLearnProcessor. 
    # For custom deps (nibabel), usually you need a custom image or install in script.
    # Here we use SKLearnProcessor as a base image and assume we can install libs or use a custom image.
    # For simplicity, we assume the base image has python and we can install what we need, 
    # OR we use a FrameworkProcessor.
    
    # NOTE: In a real prod env, build a Docker image with nibabel installed.
    # sending a requirements.txt to ScriptProcessor is possible but sometimes tricky depending on the image.
    # We will attempt to use a standard image and install at runtime in the script if needed, 
    # OR better, use a simpler approach for this PoC.
    
    # using SKLearnProcessor for simplicity, but we are running python code.
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-process"
    )
    
    # Create the Processing Step
    step_process = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/output")
        ],
        code="preprocessing.py" # This uploads our local script
    )
    
    # -------------------------------------------------------------------------
    # 3. Training Step
    # -------------------------------------------------------------------------
    # Use the TensorFlow estimator
    
    tf_estimator = TensorFlow(
        entry_point='train.py',
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        framework_version='2.3.0', # Choose appropriate version
        py_version='py37',
        hyperparameters={
            'epochs': 5,
            'batch-size': 4
        }
    )
    
    step_train = TrainingStep(
        name="TrainModel",
        estimator=tf_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="application/x-npy"
            )
        }
    )
    
    # -------------------------------------------------------------------------
    # 4. Pipeline Definition
    # -------------------------------------------------------------------------
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            input_data_uri
        ],
        steps=[step_process, step_train]
    )
    
    return pipeline

if __name__ == "__main__":
    # Create/Update pipeline
    # Note: Requires AWS credentials
    import json
    
    try:
        if aws_config.ROLE is None:
             aws_config.ROLE = sagemaker.get_execution_role()
             
        pipeline = get_pipeline(
            region=aws_config.REGION,
            role=aws_config.ROLE,
            default_bucket=aws_config.BUCKET_NAME
        )
        
        print(f"Pipeline definition: {json.loads(pipeline.definition())['steps']}")
        pipeline.upsert(role_arn=aws_config.ROLE)
        print(f"Pipeline '{pipeline.name}' created/updated successfully.")
        
    except Exception as e:
        print(f"Failed to create pipeline: {e}")
        print("Ensure you have set the Role correctly in aws_config.py")

