BUCKET_ROOT="gs://sandbox-cmle/pong"

TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.pong"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="pong_$now"

JOB_DIR=$BUCKET_ROOT"/output"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml
