#####1\pretrain-stage 1
python train_model_stage1.py

#######2\pretrain-stage 2
python train_model_stage2.py

#######3\inference
python Z_generation.py

#######4\downloadn task
python train_property_predict/regress_predict.py
