#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fast_wsi_conch

# Instance Shot = 16
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 42 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed42_1_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 42 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed42_2_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 42 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed42_4_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 42 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed42_8_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 42 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed42_16_16.txt


python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 43 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed43_1_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 43 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed43_2_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 43 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed43_4_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 43 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed43_8_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 43 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed43_16_16.txt


python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 44 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed44_1_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 44 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed44_2_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 44 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed44_4_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 44 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed44_8_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 44 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed44_16_16.txt


python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 45 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed45_1_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 45 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed45_2_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 45 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed45_4_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 45 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed45_8_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 45 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed45_16_16.txt


python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 46 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed46_1_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 46 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed46_2_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 46 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed46_4_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 46 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed46_8_16.txt
python FAST_CAMELYON_CONCH_Train_CONCH.py  --seed 46 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 1000 --batch_size 4096 > ./log/CONCHFAST_CAMELYON_forPublic_rebuttal/16_InstanceShot/Seed46_16_16.txt
