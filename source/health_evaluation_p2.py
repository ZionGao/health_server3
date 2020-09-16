import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, truncnorm, randint
import joblib


'''
数据录入-模型训练-测试输出 Interface3

调用这个health_prediction_p2.py程序的前提：
(1) 用户填写完第2页网页各项，将网页各项按照下面顺序，转换为数字后，保存到“health_evaluation.csv”中
顺序:
年龄	性别	胸腔疼痛类型	静止血压	胆固醇	空腹血>120mg/dl	静息心电图结果	最大心率	运动诱发心绞痛	ST段压低	峰值运动ST段斜率	颜色透视大血管数	铊中毒血液病
(2) 在这里写接口，接收前台给的
(3) 训练样本heart.csv在后台读入health_evaluation.csv
(4) 训练样本heart.csv在后台读入
(5) 用户点击“健康评估”箭头后触发这个程序
(6) heart.csv中 target - 0 正常，1 具有健康风险
(7) 程序运行比较慢，所以前端最好能画一个进度条，当本程序调用之后，进度条置顶即可
'''

def predict_one(test_data):
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    # 显示所有行
    # pd.set_option('display.max_rows', None)

    # 这里烦劳志强写一下，heart.csv数据文件是模型的训练样本，这里应该存在数据库里，读取出来
    train_data = pd.read_csv("./data/heart.csv", encoding='utf-8')
    # 这里烦劳志强写一下，健康预测批量测试样本.csv数据文件是用于保存测试样本的（允许用户根据上面的格式逐个单元输入）
    # test_data = pd.read_csv("./data/health_evaluation.csv", encoding='utf-8')


    cl1 = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
    cl2 = ["年龄","性别","胸腔疼痛类型","静止血压","胆固醇","空腹血>120mg/dl","静息心电图结果","最大心率","运动诱发心绞痛","ST段压低","峰值运动ST段斜率","颜色透视大血管数","铊中毒血液病","目标值"]
    dic = dict(zip(cl2,cl1))
    test_data.rename(columns=dic,inplace=True)

    # test_data.columns = train_data.columns



    all_data = pd.concat([train_data, test_data], axis=0)

    print(all_data)

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

    # Set Up a Grid of Hyperparameter Values
    # I’ll tune three hyperparameters: n_estimators, max_features, and min_samples_split
    # Define and Train the Model with Random Search
    model_params = {
        # randomly sample numbers from 4 to 204 estimators
        'n_estimators': randint(4,200),
        # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
        'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
        # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
        'min_samples_split': uniform(0.01, 0.199)
    }

    # # create random forest classifier model
    rf_model = RandomForestClassifier()
    #
    # # set up random search meta-estimator
    # # this will train 100 models over 5 folds of cross validation (500 models total)
    clf = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)

    # train the random search meta-estimator to find the best model out of 100 candidates
    model = clf.fit(x, y)
    joblib.dump(model, './model/single.model')
    # model = joblib.load('./model/single.model')
    # print winning set of hyperparameters

    #pprint(model.best_estimator_.get_params())

    x_test = all_data.iloc[train_data.shape[0]:]

    print(x_test)


    predictions_test = model.predict(x_test)

    # 这里是个接口，要烦劳志强写一下，把对测试样本的预测结果输出给网页
    # predictions_test == 1 时，有健康风险，第2个网页上红色按钮激活
    # predictions_test == 0 时，正常，第2个网页上绿色按钮激活
    return predictions_test[0]







