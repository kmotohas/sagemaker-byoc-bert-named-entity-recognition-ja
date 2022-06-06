import os
import sys
from logging import getLogger, DEBUG, StreamHandler

import hydra
from omegaconf import DictConfig

import sagemaker
from sagemaker import get_execution_role
from sagemaker.local import LocalSession
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.model import Model

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler(sys.stdout))

#sagemaker_session = LocalSession()
#sagemaker_session.config = {'local': {'local_code': True}}
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

role = get_execution_role()


@hydra.main(version_base=None, config_path='config', config_name='inference')
def test_inference(cfg: DictConfig):
    model_data = f's3://{bucket}/{cfg.training_job_name}/output/model.tar.gz'

    if cfg.container_type == 'built-in':
        # create Hugging Face Model Class
        model = HuggingFaceModel(
            model_data=model_data,        # path to your model and script
            role=role,                    # iam role with permissions to create an Endpoint
            transformers_version="4.12",  # transformers version used
            pytorch_version="1.9",        # pytorch version used
            py_version='py38',            # python version used
            entry_point=cfg.entry_point,
            source_dir=os.path.join(os.path.dirname(__file__), f'../{cfg.source_dir}'),
        )
    elif cfg.container_type == 'byoc':
        model = Model(
            image_uri=cfg.image_uri,
            model_data=model_data,
            role=role,
            entry_point=cfg.entry_point,
            source_dir=os.path.join(os.path.dirname(__file__), f'../{cfg.source_dir}'),
        )
    else:
        logger.fatal("container_type must be 'byoc' or 'built-in'")
        exit()
    
    batch_job = model.transformer(
        instance_count=cfg.instance_count,
        instance_type=cfg.instance_type,
        strategy='SingleRecord',
    )
    
    batch_job.transform(
        data=f's3://{bucket}/ner/preprocessed/test/input.jsonl',
        content_type='application/json',
        split_type='Line'
    )

if __name__ == '__main__':
    test_inference()
