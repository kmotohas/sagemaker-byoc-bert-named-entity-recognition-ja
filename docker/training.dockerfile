# https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-us-east-1.html#huggingface-us-east-1.title
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.9-transformers4.12-gpu-py38-cu111-ubuntu20.04

ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

COPY scripts/* /opt/ml/code/
RUN pip install -r /opt/ml/code/requirements.txt
ENV SAGEMAKER_PROGRAM train.py
