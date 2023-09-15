from pendulum import datetime, duration
from io import BytesIO

import pandas as pd
import torch
import requests
import pyarrow as pa
import numpy as np
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

S3_CONN_ID = "aws_default"
BUCKET = "ml-pipeline-tutorial"

def upload_to_s3(file, cache_path):
    # initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)

    # write series of bytes to s3 object in the given bucket
    with BytesIO() as buffer:
        torch.save(file, buffer)
        s3_hook.load_bytes(buffer.getvalue(), cache_path, bucket_name=BUCKET, replace=True)

def download_from_s3(cache_path):
    # initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)

    # default to Sep 10th, 2023 data by default
    data_bytes = s3_hook.get_key(key=cache_path, bucket_name=BUCKET)

    # Read the bytes of the object from the object body
    data_bytes = data_bytes.get()['Body'].read()

    # Read bytes into a buffer
    buffer = BytesIO(data_bytes)

    return buffer