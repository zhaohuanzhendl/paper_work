
![](http://latex.codecogs.com/gif.latex?\\frac{1}{1+sin(x)})  
![](http://latex.codecogs.com/gif.latex?X_{i})  

KPCA简介:  
原始空间R^m中有n个样本 x_{1},x_{2},⋅⋅⋅,x_{n}, 由这n个样本构成的数据矩阵为X，利用非线性映射函数\Phi 将原始空间X映射到高维特征空间 F={h(x)|x∈X} 中，  
假设其映射空间的维数为M (其中M远大于m)。  
将中心化后的数据记为\widetilde{\Phi(x_i)}  
数据均值计算为\overline{\Phi} = \frac{1}{n}\sum^n_{i}\Phi(x_i)  
即 \widetilde{\Phi(x_i)} = \Phi(x_i) - \overline{\Phi}

中心化的数据\widetilde{\Phi(x_i)}的协方差矩阵为   
\overline{C}=\frac{1}{n}\sum^n_{k=1}\widetilde{\Phi(x_k)}\widetilde{\Phi(x_k)}^T = \frac{1}{n}\widetilde{\Phi(X))}\widetilde{\Phi(X)}^T    
