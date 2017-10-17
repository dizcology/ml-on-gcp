BUCKET="gs://sandbox-cmle/"

TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="pong_$now"

JOB_DIR=$BUCKET$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    -- \
    --output-dir "gs://sandbox-cmle/pong_200" \
    --restore \
    #--job-dir $JOB_DIR \
    #--learning-rate 0.0005 \
    #--hidden-dims 100 100 \
    #--batch-size 1 \
    #--n-batch 60000
    
