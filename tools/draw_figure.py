import matplotlib.pyplot as plt
import numpy as np

# x_c = np.load("./update_count_count_transi_force_copy.npy")
x_c = np.load("./update_count_count_transi_force_nocopy.npy")
# x_c = x_c[::10]
x_c2 = np.load("./update_count_count_transi_force_nocopy1.npy")
y = np.arange(x_c.shape[0])+1
# 设置坐标轴的名称
# plt.xlabel('embed_index')
# plt.ylabel('update_count')

# plt.title("右侧底部标题", y=0, loc='right')
# plt.title("左侧顶部标题", y=1, loc='left')
plt.title("features std range")

# 设置坐标轴上的刻度
new_ticks = np.linspace(-1, 2, 5)

# plt.plot(y,x_c)
# plt.plot(y,x_c2[:1024])
# plt.yscale("log")
# plt.grid()

# plt.show()
plt.xlabel('features std')
plt.ylabel('features index')
# import copy
# plt2 = copy.deepcopy(plt)


# fn1=np.load("./f_transist/n1.npy").reshape(-1)
# fn2=np.load("./f_transist/n2.npy").reshape(-1)
def get_x_y_bar(fn0):
    fn0 = fn0*50
    fn0 = fn0.astype(np.uint)
    fn0 = fn0.astype(np.float32)
    fn0 = fn0/50
    fn0_unique =np.unique(fn0)

    a0 = []

    for i in np.unique(fn0_unique):
        a0.append(np.array([np.sum(fn0==i)]))
    a0 = np.concatenate(a0,axis=0)

    return fn0_unique,a0

# fn0_f=np.load("./f_transist/n0_f.npy").reshape(-1)
# fn1_f=np.load("./f_transist/n1_f.npy").reshape(-1)
# fn2_f=np.load("./f_transist/n2_f.npy").reshape(-1)

fn0=np.load("./f_transist/n0.npy").reshape(-1)
fn0_unique,a0 = get_x_y_bar(fn0=fn0)
fn0_f=np.load("./f_transist/n0_f.npy").reshape(-1)
fn0_unique_f,a0_f = get_x_y_bar(fn0=fn0_f)

# xf = np.arange(0,4096)
plt.bar(fn0_unique,a0,width= 0.012,facecolor="orange")
plt.bar(fn0_unique_f,a0_f,width= 0.012,facecolor="green")
# plt.bar(xf,fn0_f,facecolor="green")
plt.show()