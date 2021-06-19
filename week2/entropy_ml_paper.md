####Fast Information-theoretic Bayesian Optimisation    
快速信息论贝叶斯优化    
链接： https://arxiv.org/abs/1711.00673    
代码：https://github.com/rubinxin/FITBO    
这项研究主要由牛津大学工程科学系完成。这篇工作基于信息论提出了一个新的算法 FITBO。    
该算法可以直接量化两个域之间的距离大小。其基本逻辑是，FITBO 可以避免重复采样全局极小化量（global minimizer）。    
并且，该算法中核（kernel）的选择相对较多，因此性能上可能会更优。研究者已公开 FITBO 的 Matlab 代码。    

##Chi-square Generative Adversarial Network  
卡方生成对抗网络  
链接：http://proceedings.mlr.press/v80/tao18b/tao18b.pdf  
这项研究由杜克大学与复旦大学完成。为了评估真实数据和合成数据之间的差异，  
可使用分布差异度量来训练生成对抗网络（GAN）。信息论散度、积分概率度量和 Hilbert 空间差异度量是三种应用比较广泛的度量。  
在这篇论文中，研究者阐述了这三种流行的 GAN 训练标准之间的理论联系，并提出了一种全新的流程——（χ²）卡方GAN，    
其概念简单、训练稳定且能够耐受模式崩溃。这个流程可用于解决多个分布的同时匹配问题。    
此外，研究者还提出了一种重采样策略，可通过一种重要度加权机制为训练后的 critic 函数重新设定目标，从而显著提升样本质量。  
