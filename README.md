from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
np.set_printoptions(precision=5, suppress=True)

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('max_colwidth', 10000)
pd.set_option('display.width', 10000)  # 不换行

# 数据读取
file_path = "D:\\kaggle\\贷款数据\\archive\\贷款数据.csv"
data = pd.read_csv(file_path, encoding='gbk')
print(data.head())
print(data.shape)

#删除重复项
print(len(data.loan_number.unique()))
data_duplicates = data.drop_duplicates(subset = 'loan_number',keep = 'first')
print(data.shape)

#查看缺失值，缺失率达到30%，删除字段
print(data.isnull().any())
print(data_duplicates['listing_title'].isnull().sum() / len(data_duplicates)) #所有字段中listing_title有缺失值，并且缺失率约为30%

#删除缺失样本
data_duplicates = data_duplicates.dropna()
print(data_duplicates.isnull().any())
print(data_duplicates.shape)

#逾期一致化处理：为方便分析逾期情况，按逾期天数区分是否逾期，逾期天数大于0为逾期，用“1”表示，逾期天数等于0为正常还款，用“0”表示
def over_Due(x):
   if x == 0:
     return 0
   else:
     return 1
data_duplicates['over_due'] = data_duplicates['days_past_due'].apply(over_Due)

# 贷款金额分箱
bins = [0,5000,10000,15000,20000,25000,30000,35000,40000]
level = ['5K以内','5K-1W','1W-1.5W','1.5W-2W','2W-2.5W','2.5W-3W','3W-3.5W','3.5W-4W']
data_duplicates['amount_type'] = pd.cut(data_duplicates['amount_borrowed'],bins = bins,labels = level)

# 贷款利率分箱
binsr = [0.05,0.1,0.15,0.2,0.25,0.3,0.35]
levelr = ['0.05-0.1','0.1-0.15','0.15-0.2','0.2-0.25','0.25-0.3','0.3-0.35']
data_duplicates['rate_type'] = pd.cut(data_duplicates['borrower_rate'],bins = binsr,labels = levelr)
print(data_duplicates[['amount_borrowed','amount_type','borrower_rate','rate_type']])

#异常值处理:异常值是指明显偏离大多数抽样数据的数值，利用箱型图能很明显的观察到，这里主要查看1，3，9，12，15的异常值情况
plt.style.use('ggplot')
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
data_box1 = data_duplicates.iloc[:,[1]]
data_box2 = data_duplicates.iloc[:,[3]]
data_box3 = data_duplicates.iloc[:,[9]]
data_box4 = data_duplicates.iloc[:,[12]]
data_box5 = data_duplicates.iloc[:,[15]]
data_box1.boxplot()
plt.show()
data_box2.boxplot()
plt.show()
data_box3.boxplot()
plt.show()
data_box4.boxplot()
plt.show()
data_box5.boxplot()
plt.show()
#有异常值

#消除异常值
a = data_box1.quantile(0.75)
b = data_box1.quantile(0.25)
c = data_box1
c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
data_box1 = data_box1.dropna()

a = data_box2.quantile(0.75)
b = data_box2.quantile(0.25)
c = data_box1
c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
data_box2 = data_box2.dropna()

a = data_box3.quantile(0.75)
b = data_box3.quantile(0.25)
c = data_box3
c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
data_box3 = data_box3.dropna()

a = data_box4.quantile(0.75)
b = data_box4.quantile(0.25)
c = data_box4
c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
data_box4 = data_box4.dropna()

a = data_box5.quantile(0.75)
b = data_box5.quantile(0.25)
c = data_box5
c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
data_box5 = data_box5.dropna()

data_box1.boxplot()
plt.show()
data_box2.boxplot()
plt.show()
data_box3.boxplot()
plt.show()
data_box4.boxplot()
plt.show()
data_box5.boxplot()
plt.show()

#数据可视化分析

#整体贷款情况
total = data_duplicates['over_due'].count()
bad = data_duplicates['over_due'].sum()
good = total - bad
values = [good,bad]
plt.figure(figsize=(6,6))
label = ['正常还款','逾期']
explode = [0.01,0.01]
plt.pie(values,explode = explode,labels = label,autopct='%1.1f%%')
plt.show()

#逾期原因分析:根据字段，构建模型需要解决这些问题：逾期用户的借款金额类型、借款期限、贷款利率、评级、借款用途各自如何分布。这些维度之间的关联性如何？
over_due_fre_am = data_duplicates.groupby('amount_type').agg({'over_due': lambda x: x.sum()/total}) # 注意分母是total，而非x.sum()，否则求出来的是该组里的逾期率
over_due_fre_te = data_duplicates.groupby('term').agg({'over_due': lambda x: x.sum()/total})
over_due_fre_gr = data_duplicates.groupby('grade').agg({'over_due': lambda x: x.sum()/total})
over_due_fre_ra = data_duplicates.groupby('rate_type').agg({'over_due': lambda x: x.sum()/total})
over_due_fre_li = data_duplicates.groupby('listing_title').agg({'over_due': lambda x: x.sum()/total}).sort_values(by = 'over_due',ascending=False)
ax1 = plt.subplot(221)
over_due_fre_am.plot(kind = 'bar',ax = ax1)
plt.title('贷款金额逾期率')
ax2 = plt.subplot(222)
over_due_fre_te.plot(kind = 'bar',ax = ax2)
plt.title('分期期别逾期率')
ax3 = plt.subplot(223)
over_due_fre_gr.plot(kind = 'bar',ax = ax3)
plt.title('各等级逾期率')
ax4 = plt.subplot(224)
over_due_fre_ra.plot(kind = 'bar',ax = ax4)
plt.title('分期利率逾期率')
plt.xticks(rotation = 360)
over_due_fre_li.plot(kind = 'bar')
plt.title('贷款用途逾期率')
plt.show()

#逾期类型为债务合并人群中，是不是5K-1W区间贷款金额最高？
data = data_duplicates[data_duplicates['listing_title'] == 'debt_consolidation']
data = data[data['over_due'] == 1]
total = data['over_due'].sum()
con_fre = data.groupby('amount_type').agg({'over_due':lambda x:x.count()/total}).sort_values(by = 'over_due',ascending=True).plot(kind = 'barh')
plt.title('债务合并')
plt.show()
