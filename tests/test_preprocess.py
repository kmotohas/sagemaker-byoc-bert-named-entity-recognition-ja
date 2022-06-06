import os
import sys
from logging import getLogger, DEBUG, StreamHandler

import hydra
from omegaconf import DictConfig

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.local import LocalSession
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import Processor
from sagemaker.processing import ProcessingInput, ProcessingOutput

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler(sys.stdout))

#s3 = boto3.client('s3')
#sagemaker_session = LocalSession()
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
#sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
#role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
role = get_execution_role()

@hydra.main(version_base=None, config_path='config', config_name='preprocess')
def test_preprocess(cfg: DictConfig):
    processor = None
    if cfg.container_type == 'built-in':
        # built-in container
        processor = HuggingFaceProcessor(
            role=role,
            instance_count=1,
            instance_type=cfg.instance_type,
            transformers_version='4.12',
            pytorch_version='1.9',
            py_version='py38',
        )
        processor.run(
            code=cfg.entry_point,  # built-in container
            source_dir=os.path.join(os.path.dirname(__file__), f'../{cfg.source_dir}'),
            inputs=[
                ProcessingInput(
                    input_name='raw',
                    source=f's3://{bucket}/ner/raw',
                    destination='/opt/ml/processing/input/raw',
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='preprocessed',
                    source='/opt/ml/processing/output/data',
                    destination=f's3://{bucket}/ner/preprocessed',
                    s3_upload_mode='EndOfJob',
                )
            ],
        )
    elif cfg.container_type == 'byoc':
        # byoc
        processor = Processor(
            image_uri=cfg.image_uri,
            role=role,
            instance_count=1,
            instance_type=cfg.instance_type,
            entrypoint=['python3', f'/opt/ml/code/{cfg.entry_point}'],
        )
        processor.run(
            inputs=[
                ProcessingInput(
                    input_name='raw',
                    source=f's3://{bucket}/ner/raw',
                    destination='/opt/ml/processing/input/raw',
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='preprocessed',
                    source='/opt/ml/processing/output/data',
                    destination=f's3://{bucket}/ner/preprocessed',
                    s3_upload_mode='EndOfJob',
                )
            ],
        )
    else:
        logger.fatal("container_type must be 'byoc' or 'built-in'")
        exit()

if __name__ == '__main__':
    test_preprocess()
