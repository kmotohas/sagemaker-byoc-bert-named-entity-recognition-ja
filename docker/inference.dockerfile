# https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-us-east-1.html#huggingface-us-east-1.title
# GPU
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.9-transformers4.12-gpu-py38-cu111-ubuntu20.04
# CPU
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.9-transformers4.12-cpu-py38-ubuntu20.04

ENV PATH="/opt/ml/code:${PATH}"
COPY scripts/* /opt/ml/code/
RUN pip install -r /opt/ml/code/requirements.txt
