#######1000个分子的31种属性进行约束生成，两个同步进行生成（并行）
python Z_generate_all_sh.py
python Z_generate_all_sh_2.py

#######单属性约束分子生成#######################
python Z_generate_single_sh.py
#mean_rmse 0.2560632;mean_uni 0.8011518;mean_val 0.9936659999999999;mean_nov 0.9984130745354832
#result_rmse_lt_04_08_with_smiles_1... 合并成all_1000PV_test.csv

####随机多属性约束分子生成#####################
python Z_generate_multi_sh.py

#####无属性约束生成###########################
python Z_generate_ddd_fudion_RDM_5_noise.py

#####外推到训练属性范围外生成
python Z_generate_ddd_fudion_RDM_5_noise_out.py

#####案例分析
python Z_generate_ddd_fudion_RDM_5_noise_case.py

