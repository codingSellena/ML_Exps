import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据
iterations = [0, 1, 2, 3, 4, 5]
avg_path_length = [66.65842827, 64.4191549, 62.10269559, 60.51860523, 59.94242909, 59.47513137]

# 绘图
plt.figure(figsize=(6, 4))
plt.plot(iterations, avg_path_length, marker='o', linestyle='-', color='b')

# 在每个数据点的右上方标注具体数值
for i, txt in enumerate(avg_path_length):
    plt.text(iterations[i] + 0.1, avg_path_length[i] + 0.1, f'{txt:.2f}', ha='left', va='bottom', fontsize=10, color='black')

# 设置标题和标签
plt.title("迭代次数与平均路径长度的关系")
plt.xlabel("迭代次数")
plt.ylabel("平均路径长度")

# 显示网格
plt.grid(True)

plt.savefig("avg_path_length.png",dpi=600)
# 显示图表
plt.show()
