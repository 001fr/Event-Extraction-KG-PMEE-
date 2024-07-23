# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 加载.npy文件
# visible_matrix = np.load('D:/1/visible_matrix_33.npy')
#
# plt.figure(figsize=(12, 12))
# fig, ax = plt.subplots(figsize=(12, 12))
# sns.heatmap(visible_matrix, cmap='Reds', square=True, vmin=0, vmax=1, annot=False, linewidth=1, ax=ax)
# ax.tick_params(axis='both', which='major', labelsize=5)
# plt.title('Visible Matrix', fontsize=16)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
# 设置全局字体为Times
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
# 指定中文楷体字体
chinese_font = FontProperties(fname=r"c:\Windows\Fonts\simkai.ttf")
# 加载.npy文件
visible_matrix = np.load('D:/1/visible_matrix_10.npy')

# 定义句子
# sentence = "部署事件：一架新的A330-MRTT英国加油机“凤凰”部署到卡塔尔。次日，即执行首次对临时联盟战机和约旦起飞的法军阵风法国战斗机的加油任务。"
sentence = "一架新的A330-MRTT英国加油机“凤凰”部署到卡塔尔"
# 创建字符列表
char_list = list(sentence)
# plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(visible_matrix, cmap='Reds', square=True, vmin=0, vmax=1, annot=False, linewidth=1, ax=ax,
           xticklabels=np.arange(len(char_list)), yticklabels=np.arange(len(char_list)))

# 设置坐标轴标签大小
ax.tick_params(axis='both', which='major', labelsize=4)

# 设置坐标轴标签为中文
ax.set_xticklabels(char_list, fontproperties=chinese_font, fontsize=18,rotation=0)
ax.set_yticklabels(char_list, fontproperties=chinese_font, fontsize=18,rotation=0)

# 去除旁边的空白区域
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("D:/图.png", dpi=300, format="png",bbox_inches='tight')
plt.show()

