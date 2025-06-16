#!/bin/bash

python Renal_MIL_Adapter.py  --seed 42 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed42_1_16.txt &
python Renal_MIL_Adapter.py  --seed 42 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed42_2_16.txt &
python Renal_MIL_Adapter.py  --seed 42 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed42_4_16.txt &
python Renal_MIL_Adapter.py  --seed 42 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed42_8_16.txt &
python Renal_MIL_Adapter.py  --seed 42 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed42_16_16.txt &

wait

python Renal_MIL_Adapter.py  --seed 43 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed43_1_16.txt &
python Renal_MIL_Adapter.py  --seed 43 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed43_2_16.txt &
python Renal_MIL_Adapter.py  --seed 43 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed43_4_16.txt &
python Renal_MIL_Adapter.py  --seed 43 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed43_8_16.txt &
python Renal_MIL_Adapter.py  --seed 43 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed43_16_16.txt &

wait

python Renal_MIL_Adapter.py  --seed 44 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed44_1_16.txt &
python Renal_MIL_Adapter.py  --seed 44 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed44_2_16.txt &
python Renal_MIL_Adapter.py  --seed 44 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed44_4_16.txt &
python Renal_MIL_Adapter.py  --seed 44 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed44_8_16.txt &
python Renal_MIL_Adapter.py  --seed 44 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed44_16_16.txt &

wait

python Renal_MIL_Adapter.py  --seed 45 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed45_1_16.txt &
python Renal_MIL_Adapter.py  --seed 45 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed45_2_16.txt &
python Renal_MIL_Adapter.py  --seed 45 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed45_4_16.txt &
python Renal_MIL_Adapter.py  --seed 45 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed45_8_16.txt &
python Renal_MIL_Adapter.py  --seed 45 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed45_16_16.txt &

wait

python Renal_MIL_Adapter.py  --seed 46 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed46_1_16.txt &
python Renal_MIL_Adapter.py  --seed 46 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed46_2_16.txt &
python Renal_MIL_Adapter.py  --seed 46 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed46_4_16.txt &
python Renal_MIL_Adapter.py  --seed 46 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed46_8_16.txt &
python Renal_MIL_Adapter.py  --seed 46 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances 1.0 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/RENAL_forPublic/Seed46_16_16.txt &
