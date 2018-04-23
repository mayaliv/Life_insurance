import pandas as pd
from sklearn.preprocessing import Imputer
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import re
import mord


#Define the file names
training_data_file="train.csv"
real_data_file="test.csv"

#import the dataset
train_data=pd.read_csv(training_data_file)
real_data=pd.read_csv(real_data_file)
print(train_data.shape)

#define the target
target=train_data['Response']

#preprocess the catagorical data
cat_string='Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41'
list_catagorical= [x for x in re.compile('\s*[,|\s+]\s*').split(cat_string)]
catagorical_data=train_data[list_catagorical]
catagorical_data = pd.get_dummies(catagorical_data, columns = list_catagorical )

list_continous= ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']

continous_data=train_data[list_continous]

# implementing mean strategy to replace 'NaN'
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_data[list_continous])
features1 = imp.transform(continous_data)
continous_data = pd.DataFrame(features1, columns=list_continous)
print(continous_data.head())
print(continous_data.shape)
print(catagorical_data.head())

#normelizing the data
features=continous_data
scaler = MinMaxScaler().fit(features.values)
#scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X = pd.DataFrame(features, columns = list_continous)

#merge text and numeric data
X = pd.concat([X, catagorical_data], axis=1)

# # Split using to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.33,random_state=22)
#
#


model=mord.LogisticIT(alpha=0.1, verbose=0, max_iter=10000)
results=model.fit(X_train,y_train)
pred=model.predict(X_test)
score=model.score(X_test, y_test)
print('LogisticIT')
print(score)


