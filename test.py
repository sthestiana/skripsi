import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


featureNames = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

df = pd.read_csv('abalone.data', header=None, names=featureNames)
df.head()

f, axes = plt.subplots(3, 3, figsize=(20,40))
sns.distplot(df["Length"], ax=axes[0][0])
sns.distplot(df["Diameter"], ax=axes[0][1])
sns.distplot(df["Height"], ax=axes[0][2])
sns.distplot(df["Whole weight"], ax=axes[1][0])
sns.distplot(df["Shucked weight"], ax=axes[1][1])
sns.distplot(df["Viscera weight"], ax=axes[1][2])
sns.distplot(df["Shell weight"], ax=axes[2][1])
#plt.show()

def removeOutlier(df, col_name, threshold, upper=True):    
    if(upper==True):
        df = df.drop(df[(df[col_name] > threshold)].index)
    else:
        df = df.drop(df[(df[col_name] < threshold)].index)
    return df
    
df = removeOutlier(df, 'Height', 0.25)

df["Length"] = np.square(df["Length"])
df["Diameter"] = np.square(df["Diameter"])
df["Whole weight"] = np.sqrt(df["Whole weight"])
df["Shucked weight"] = np.sqrt(df["Shucked weight"])
df["Viscera weight"] = np.sqrt(df["Viscera weight"])
df["Shell weight"] = np.sqrt(df["Shell weight"])

f, axes = plt.subplots(3, 3, figsize=(15,30))
sns.distplot(df["Length"], ax=axes[0][0])
sns.distplot(df["Diameter"], ax=axes[0][1])
sns.distplot(df["Height"], ax=axes[0][2])
sns.distplot(df["Whole weight"], ax=axes[1][0])
sns.distplot(df["Shucked weight"], ax=axes[1][1])
sns.distplot(df["Viscera weight"], ax=axes[1][2])
sns.distplot(df["Shell weight"], ax=axes[2][1])
#plt.show()


X = df
y = X.pop('Rings')
X.pop('Sex')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))

from sklearn.decomposition import PCA
pca = PCA().fit(X)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')

pca = PCA(1)
pca.fit(X_train)
new_X_train = pca.transform(X_train)
new_X_test = pca.transform(X_test)
#model = MLPClassifier(max_iter=1000, activation="logistic", hidden_layer_sizes=(8,8))
model = LinearRegression()
model.fit(new_X_train, y_train)

print(model.score(new_X_test, y_test))
print(model.score(new_X_train, y_train))

y_pred = model.predict(new_X_test)
# Plot outputs
plt.scatter(new_X_test, y_test,  color='black')
plt.plot(new_X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()