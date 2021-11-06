import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE=24 #一条dna（解的二进制）的长度
POP_SIZE=100#种群（解空间）的大小
CROSSOVER_RATE = 0.6#交叉概率
MUTATION_RATE = 0.005#变异概率
N_GENERATIONS = 100#迭代次数
X_BOUND = [-32, 32]#解一的范围
Y_BOUND = [-32, 32]#解二的范围

def F(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 2
    sum_sq_term = -a * np.exp(-b * np.sqrt((x * x + y * y) / d))
    cos_term = -np.exp((np.cos(c * x) + np.cos(c * y)) / d)
    z = a + np.exp(1) + sum_sq_term + cos_term
    return z

#转为十进制
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 二进制中的奇数列表示X
    y_pop = pop[:, ::2]  # 二进制中的偶数列表示y
    # print(x_pop.shape)
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

def get_fitness(pop):#适应度（误差）函数，pop是种群（可能解的集合）
    x, y = translateDNA(pop)
    pred = F(x, y)
    # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],
    # 最后在加上一个很小的数防止出现为0的适应度
    return -(pred - np.max(pred)) + 1e-3

def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):#交叉和变异，交叉概率是0.8
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop

def mutation(child, MUTATION_RATE=0.003):#变异，变异概率设为0.003
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=(fitness) / (fitness.sum()))
    #choice里的最后一个参数p，参数p描述了从np.arange(POP_SIZE)里选择每一个元素的概率，概率越高约有可能被选中，最后返回被选中的个体即可。
    return pop[idx]


def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)  # 返回100个在区间X_BOUND的点，种群大小设为100
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)  # 创建网格点矩阵
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-30, 30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


def print_info(pop):
    fitness = get_fitness(pop)  # 适应度
    # print("所有适应度",fitness)
    max_fitness_index = np.argmax(fitness)  # 返回最大值对应的索引
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    # print(x)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    print("最小值为：", F(x[max_fitness_index], y[max_fitness_index]))


fig = plt.figure()
ax = Axes3D(fig)
plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
plot_3d(ax)

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)#产生固定shape的随机矩阵
print('测试',pop)
y_array=[]
x_array=[]

for i in range(N_GENERATIONS):  # 迭代N代
    x, y = translateDNA(pop)#将矩阵转为十进制坐标

    if 'sca' in locals():#函数返回全部局部变量
        sca.remove()

    sca = ax.scatter(x, y, F(x, y), c='black', marker='o')#绘制散点图
    #print(x,y)
    plt.show()
    plt.pause(0.1)
    pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
    fitness = get_fitness(pop)

    max_fitness_index = np.argmax(fitness)  # 返回fitness最大值对应的索引
    x_best=x[max_fitness_index]#获得fitness最大值时的坐标一
    y_best=y[max_fitness_index]#获得fitness最大值时的坐标二
    z_best=F(x_best,y_best)#将坐标一二带入求得最优解
    x_array.append(i+1)
    y_array.append(z_best)
    print("第%d次迭代:"%i,z_best)#输出每轮迭代中最优解

    pop = select(pop, fitness)  # 选择生成新的种群，然后继续迭代

    #print(fitness)
# plt.ioff()#关闭交互模式

plt.plot(x_array,y_array)
plt.show()

print_info(pop)