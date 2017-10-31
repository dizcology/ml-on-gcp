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
    --output-dir "gs://sandbox-cmle/pong_200_noop_overlay_split1" \
    --learning-rate 0.002 \
    --allow-noop \
    --beta 0.01 \
    --gamma 0.95 \
    --decay 0.95 \
    --overlay
    #--restore
    #--restore
    #--restore \
    #--restore \
    #--allow-noop \
    #--beta 0.1
    #--job-dir $JOB_DIR \
    #--hidden-dims 100 100 \
    #--batch-size 1 \
    #--n-batch 60000 \
    
# python trainer/task.py --render --dry-run --restore --allow-noop --n-batch 1 --batch-size 1 --output-dir "gs://sandbox-cmle/pong_200_noop_overlay1"  (or smoother, smooth)

# python trainer/task.py --render --dry-run --restore --allow-noop --output-dir "gs://sandbox-cmle/pong_100_100_noop_smoothie" --hidden-dims 100 100

# python trainer/task.py --render --dry-run --restore --output-dir "gs://sandbox-cmle/pong_200"

# tensorboard --logdir="gs://sandbox-cmle/pong_200_noop_overlay1" (or smooth, smoother, or smoothie) 