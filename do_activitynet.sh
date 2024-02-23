collection=activitynet
visual_feature=i3d
device_ids=$1
exp_id=$2
ssn=$3
consistency_weight=$4
hallucination_weight=$5
root_path=$6
support_ckpt_filepath=$7
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --device_ids $device_ids --consistency_weight $consistency_weight \
                    --support_set_number $ssn --hallucination_weight $hallucination_weight \
                    --support_ckpt_filepath $support_ckpt_filepath
