from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# 设置参数
mu = 0        # 共同均值
sigma_fat = 2 # 矮胖分布的标准差
sigma_thin = 0.5 # 高瘦分布的标准差

# 生成x轴数据（覆盖两个分布的合理范围）
x = np.linspace(-6, 6, 500)

# 计算两个分布的概率密度
y_fat = norm.pdf(x, mu, sigma_fat)
y_thin = norm.pdf(x, mu, sigma_thin)

# 绘制对比图
plt.figure(figsize=(10, 5))
plt.plot(x, y_fat, 'b-', linewidth=5)
plt.plot(x, y_thin, 'r--', linewidth=5)
# plt.plot(x, y_thin, 'r--', linewidth=2, label=f'高瘦分布 (σ={sigma_thin})')


# 美化图形
# plt.title("正态分布形态对比 (相同均值μ=0)")
# plt.xlabel("数值")
# plt.ylabel("概率密度")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-6, 6)  # 固定显示范围
plt.tight_layout()
plt.show()