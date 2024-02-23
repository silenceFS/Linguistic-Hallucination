collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
margin=0.1
device_ids=$1
support_set_number=$2
consistency_weight=$3
hallucination_weight=$4
exp_id=$5
root_path=$6
support_ckpt_filepath=$7
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --dset_name $collection --exp_id $exp_id \
                    --q_feat_size $q_feat_size --margin $margin --device_ids $device_ids \
                    --support_set_number $support_set_number \
                    --consistency_weight $consistency_weight \
                    --hallucination_weight $hallucination_weight \
                    --root_path $root_path --support_ckpt_filepath $support_ckpt_filepath
                    # --bsz 32
                    # --train_support
