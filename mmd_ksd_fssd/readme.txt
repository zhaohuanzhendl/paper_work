1. mmd_ksd.ipynb 文件
  给定不同均值 验证MMD/KSD检测敏感度趋势变化曲线


2. mmd_ksd_impl.py 文件 实现 MMD/KSD 两个算法的基于numpy的原始实现
  a. MMDTEST 参考 wittawatj / interpretable-test, 抽取去MMD核心计算流程, 去除框架等其他无关引入， 纯计算MMD^2
  b. KSD 参考 wittawatj/gof， 剥离框架内容 单纯计算ksd结果


3. test_ex2.py 改造自ex/ex2_prob_params.py 

测试FSSD 在不同样本数，不同维度, 使用高斯核 方差不同的情况下 结果如下: 
rej_dict_opt 表示 使用J=5, 使用opt后的 FSSD 拒绝H_0的情况 字典k: 维度d, 字典v: 测试次数中拒绝的个数  
rej_dict_med 表示 使用J=5, 不使用opt后的 FSSD 拒绝H_0的情况 字典k: 维度d, 字典v: 测试次数中拒绝的个数   

n_samples: 4000
reps = 50
('rej_dict_opt:', {1: 50, 5: 48, 10: 29, 15: 24}) 
('rej_dict_med:', {1: 50, 5: 47, 10: 35, 15: 12})

n_samples: 2000 
reps = 50
('rej_dict_opt:', {1: 50, 5: 38, 10: 20, 15: 21}) 
('rej_dict_med:', {1: 50, 5: 43, 10: 16, 15: 7})

n_samples: 2000  
reps = 100
('rej_dict_opt:', {1: 100, 5: 73, 10: 45, 15: 42})
('rej_dict_med:', {1: 100, 5: 86, 10: 32, 15: 15})

n_samples: 4000  
reps = 100
('rej_dict_opt:', {1: 100, 5: 93, 10: 63, 15: 49}) 
('rej_dict_med:', {1: 100, 5: 95, 10: 64, 15: 26})
