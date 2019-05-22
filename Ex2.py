import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("kc_house_data.csv")
# data pre-process
data = data.fillna(0)
data = data.loc[data['id'] != 0]
data = data.loc[data['date'] != 0]
data = data.loc[data['price'] > 0]
data = data.loc[data['zipcode'] > 0]
data = data.loc[data['floors'] > 0]
data = data.loc[data['sqft_living'] > 0]
data = data.loc[data['sqft_lot'] > 0]
data = data.loc[data['sqft_above'] > 0]
data = data.loc[data['yr_built'] > 0]
data = data.loc[data['lat'] > 0]
data = data.loc[data['long'] < 0]
data = data.loc[data['sqft_living15'] > 0]
data = data.loc[data['sqft_lot15'] > 0]
# create new categorical features
data.loc[data.sqft_basement > 0, 'isBasment'] = 1
data.loc[data.sqft_basement == 0, 'isBasment'] = 0
data.loc[data.yr_renovated > 0, 'isRenovated'] = 1
data.loc[data.yr_renovated == 0, 'isRenovated'] = 0
zip_code = pd.get_dummies(data['zipcode'])
data = pd.concat([data, zip_code], axis=1)
# remove non categorical features
data = data.drop(['zipcode', 'id', 'date'], axis=1)
# data.to_csv("kc_house_data_edited.csv")

y = data['price']
x = data.loc[:, data.columns != 'price']

# question 6
for feat in x.keys():
    print(feat, " - ", np.abs(y.corr(x[feat])))

# question 7
features = x.keys()[0:19]
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        print(features[i], " - ", features[j], " - ", np.abs(x[features[i]].corr(x[features[j]])))
x = x.T
u, s, vh = np.linalg.svd(x @ x.T)
print(s)


# question 8
def perform_regression(percentage: float):
    houses_number, _number = data.shape
    train_num = int(houses_number * percentage)
    test_num = houses_number - train_num
    train = data.sample(train_num, random_state=1)
    test = data.drop(train.index)
    y_train = train['price']
    x_train = train.loc[:, data.columns != 'price'].T
    y_test = test['price']
    x_test = test.loc[:, data.columns != 'price'].T
    w = np.linalg.pinv(x_train).T @ y_train
    train_err_rate = (np.linalg.norm(x_train.T @ w - y_train)) ** 2 / train_num
    test_err_rate = (np.linalg.norm(x_test.T @ w - y_test)) ** 2 / test_num
    return train_err_rate, test_err_rate


train_errors = []
test_errors = []
samples = np.arange(1, 100) / 100
for i in samples:
    train_err, test_err = perform_regression(i)
    train_errors.append(train_err)
    test_errors.append(test_err)
plt.plot(samples, train_errors, label='train_errors')
plt.plot(samples, test_errors, label='test_errors')
plt.xlabel('train sample %')
plt.ylabel('logarithmic loss')
plt.yscale('log')
plt.legend()
plt.show()
