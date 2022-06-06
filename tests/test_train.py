import os
import sys
from logging import getLogger, DEBUG, StreamHandler

import hydra
from omegaconf import OmegaConf, DictConfig

import sagemaker
from sagemaker import get_execution_role
from sagemaker.local import LocalSession
from sagemaker.huggingface import HuggingFace
from sagemaker.estimator import Estimator

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler(sys.stdout))

#sagemaker_session = LocalSession()
#sagemaker_session.config = {'local': {'local_code': True}}
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
# For local training a dummy role will be sufficient
#role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
role = get_execution_role()

@hydra.main(version_base=None, config_path='config', config_name='train')
def test_train(cfg: DictConfig):
    # hyperparameters, which are passed into the training job
    hyperparameters = OmegaConf.to_container(cfg)['hyperparameters']
    num_labels = 2 * hyperparameters['num_entity_type'] + 1
    hyperparameters['num_labels'] = num_labels
    
    # s3 uri where our checkpoints will be uploaded during training
    job_name = cfg.job_name
    #checkpoint_s3_uri = f's3://{bucket}/{job_name}/checkpoints'

    estimator = None
    if cfg.container_type == 'built-in':
        estimator = HuggingFace(
            entry_point=cfg.entry_point,
            source_dir=os.path.join(os.path.dirname(__file__), f'../{cfg.source_dir}'),
            instance_type=cfg.instance_type,
            instance_count=cfg.instance_count,
            base_job_name=job_name,
            #checkpoint_s3_uri=checkpoint_s3_uri,
            #use_spot_instances=True,
            #max_wait=7200, # This should be equal to or greater than max_run in seconds'
            #max_run=3600, # expected max run in seconds
            role=role,
            transformers_version='4.12',
            pytorch_version='1.9',
            py_version='py38',
            hyperparameters=hyperparameters,
        )
    elif cfg.container_type == 'byoc':
        estimator = Estimator(
            image_uri=cfg.image_uri,
            entry_point=cfg.entry_point,
            source_dir=os.path.join(os.path.dirname(__file__), f'../{cfg.source_dir}'),
            instance_type=cfg.instance_type,
            instance_count=cfg.instance_count,
            base_job_name=job_name,
            #checkpoint_s3_uri=checkpoint_s3_uri,
            #use_spot_instances=True,
            #max_wait=7200, # This should be equal to or greater than max_run in seconds'
            #max_run=3600, # expected max run in seconds
            role=role,
            hyperparameters=hyperparameters,
        )
    else:
        logger.fatal("container_type must be 'byoc' or 'built-in'")
        exit()

    
    # starting the train job with our uploaded datasets as input
    estimator.fit({
        'train': f's3://{bucket}/ner/preprocessed/train',
        'valid': f's3://{bucket}/ner/preprocessed/validation',
    })

if __name__ == '__main__':
    test_train()
    