import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, truncnorm, randint
from pprint import pprint


'''
数据录入-模型训练-测试输出 Interface3

调用这个health_prediction.py程序的前提：
(1) 用户填写完健康预测批量测试样本.csv，且将健康预测批量测试样本.csv上传至服务器
(2) 训练样本heart.csv在后台读入
(3) 用户点击第3页“健康评估”箭头后触发这个程序
(4) target - 0 正常，1 具有健康风险
(5) 程序运行比较慢，所以前端最好能画一个进度条，当本程序调用之后，进度条置顶即可
'''
def predict_batch(test_data):
    # 这里烦劳志强写一下，heart.csv数据文件是模型的训练样本，这里应该存在数据库里，读取出来
    train_data = pd.read_csv("./data/heart.csv", encoding='utf-8')
    # 这里烦劳志强写一下，健康预测批量测试样本.csv数据文件是用于保存测试样本的（允许用户根据上面的格式逐个单元输入）
    # test_data = pd.read_csv("../data/健康预测批量测试样本.csv", encoding='utf-8')

    test_data.columns = train_data.columns
    all_data = pd.concat([train_data, test_data], axis=0)
    all_data = all_data.drop(['target'], axis=1)

    #检查缺失项
    ind_nan = all_data.isna().any()
    if(sum(ind_nan)):
        # 这里是个接口，要烦劳志强写一下，就是如果用户输入测试样本时有缺失值("健康预测批量测试样本.csv"),就弹出对话框提示用户重新输入
        print('输入存在空缺项，请核对')
    else:
        print('OK')

    # 创建哑变量
    all_data = pd.get_dummies(all_data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

    # 数值型属性数据归一化
    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    all_data[columns_to_scale] = standardScaler.fit_transform(all_data[columns_to_scale])

    # 训练模型
    # Prepare data to train
    x = all_data.iloc[:train_data.shape[0]]
    y = train_data['target']

    # # Set Up a Grid of Hyperparameter Values
    # # I’ll tune three hyperparameters: n_estimators, max_features, and min_samples_split
    # # Define and Train the Model with Random Search
    # model_params = {
    #     # randomly sample numbers from 4 to 204 estimators
    #     'n_estimators': randint(4,200),
    #     # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    #     'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    #     # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    #     'min_samples_split': uniform(0.01, 0.199)
    # }
    #
    # # create random forest classifier model
    # rf_model = RandomForestClassifier()
    #
    # # set up random search meta-estimator
    # # this will train 100 models over 5 folds of cross validation (500 models total)
    # clf = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)

    # train the random search meta-estimator to find the best model out of 100 candidates
    # model = clf.fit(x, y)
    model = joblib.load('./model/single.model')
    # print winning set of hyperparameters
    # from pprint import pprint
    # pprint(model.best_estimator_.get_params())

    x_test = all_data.iloc[train_data.shape[0]:]
    predictions_test = model.predict(x_test)
    label = test_data.loc[:,'target'].values

    # 这里是个接口，要烦劳志强写一下，把对测试样本的预测结果输出给网页
    # print(predictions_test) #输出出去
    # print(label)
    # print(list(zip(predictions_test,label)))

    return list(map(lambda x:str(x),list(predictions_test))),list(map(lambda x:str(x),list(label)))



