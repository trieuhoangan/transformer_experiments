BASIC_SETTING_PATH="$HOME/pascal/NMT/script/basic_setting.sh"

source custom_variable.sh
source $BASIC_SETTING_PATH

echo eval $TRAN_CMD $arch 2>&1 | tee $CKPTS/train.log
eval $TRAN_CMD $arch 2>&1 | tee $CKPTS/train.log