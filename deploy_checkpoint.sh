#!/bin/bash

set -e

S3_BUCKET_NAME=s3://wordle-rl/checkpoints
DEPLOY_CHECKPOINT_NAME=a2c_deployed.ckpt

aws s3 cp $1 $S3_BUCKET_NAME/$1
aws s3 cp $S3_BUCKET_NAME/$1 $S3_BUCKET_NAME/$DEPLOY_CHECKPOINT_NAME
