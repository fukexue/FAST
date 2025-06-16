#!/bin/bash

# Instance Shot = 16
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed42_1_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed42_2_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed42_4_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed42_8_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed42_16_16.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed43_1_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed43_2_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed43_4_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed43_8_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed43_16_16.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed44_1_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed44_2_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed44_4_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed44_8_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed44_16_16.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed45_1_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed45_2_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed45_4_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed45_8_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed45_16_16.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 1  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed46_1_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 2  --num_instance_shot 16 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed46_2_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 4  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed46_4_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 8  --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed46_8_16.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 16 --num_instance_shot 16 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/16_InstanceShot/Seed46_16_16.txt &
wait



# Instance Shot = 4
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 1  --num_instance_shot 4 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed42_1_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 2  --num_instance_shot 4 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed42_2_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 4  --num_instance_shot 4 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed42_4_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 8  --num_instance_shot 4 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed42_8_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 16 --num_instance_shot 4 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed42_16_4.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 1  --num_instance_shot 4 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed43_1_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 2  --num_instance_shot 4 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed43_2_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 4  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed43_4_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 8  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed43_8_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 16 --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed43_16_4.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 1  --num_instance_shot 4 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed44_1_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 2  --num_instance_shot 4 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed44_2_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 4  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed44_4_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 8  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed44_8_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 16 --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed44_16_4.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 1  --num_instance_shot 4 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed45_1_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 2  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed45_2_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 4  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed45_4_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 8  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed45_8_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 16 --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed45_16_4.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 1  --num_instance_shot 4 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed46_1_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 2  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed46_2_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 4  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed46_4_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 8  --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed46_8_4.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 16 --num_instance_shot 4 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/4_InstanceShot/Seed46_16_4.txt &
wait



# Instance Shot = 64
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 1  --num_instance_shot 64 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed42_1_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 2  --num_instance_shot 64 --downsample_neg_instances 1.0    --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed42_2_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 4  --num_instance_shot 64 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed42_4_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 8  --num_instance_shot 64 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed42_8_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 42 --num_bag_shot 16 --num_instance_shot 64 --downsample_neg_instances -2000  --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed42_16_64.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 1  --num_instance_shot 64 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed43_1_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 2  --num_instance_shot 64 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed43_2_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 4  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed43_4_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 8  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed43_8_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 43 --num_bag_shot 16 --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed43_16_64.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 1  --num_instance_shot 64 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed44_1_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 2  --num_instance_shot 64 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed44_2_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 4  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed44_4_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 8  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed44_8_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 44 --num_bag_shot 16 --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed44_16_64.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 1  --num_instance_shot 64 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed45_1_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 2  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed45_2_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 4  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed45_4_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 8  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed45_8_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 45 --num_bag_shot 16 --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed45_16_64.txt &
wait

python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 1  --num_instance_shot 64 --downsample_neg_instances 1.0   --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed46_1_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 2  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed46_2_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 4  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed46_4_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 8  --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed46_8_64.txt &
python FAST_CAMELYON_CLIP_Train.py  --seed 46 --num_bag_shot 16 --num_instance_shot 64 --downsample_neg_instances -2000 --lr_keys 0.001 --lr_values 0.01 --epochs 20000 --batch_size 4096 > ./log/CAMELYON_TextBranchLearnable/64_InstanceShot/Seed46_16_64.txt &
wait