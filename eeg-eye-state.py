#%%
from pandas_profiling import ProfileReport
import pandas as pd
# %%
df_eye = pd.read_csv('eye-state.csv')
df_eye.head()
# %%
profile = ProfileReport(df_eye, title='Eye State')
# %%
profile
# %%
X = df_eye.drop('Class', axis=1)
y = df_eye['Class']
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=33)

# %%
df_eye.info()
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

dummy = DummyClassifier()

dummy.fit(X_train, y_train)

dummypred = dummy.predict(X_test)

print(classification_report(y_test, dummypred))

#%%
from sklearn.ensemble import RandomForestClassifier

model_forest = RandomForestClassifier(random_state=5)

model_forest.fit(X_train, y_train)

y_pred = model_forest.predict(X_test)

print(classification_report(y_test, y_pred, digits=5))
# %%
from sklearn.svm import SVC

model_svm = SVC(kernel='rbf', random_state=5)

model_svm.fit(X_train, y_train)

y_pred_svm = model_svm.predict(X_test)

print(classification_report(y_test, y_pred_svm))
#%%
from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier(random_state=5)

model_mlp.fit(X_train, y_train)

y_pred_mlp = model_mlp.predict(X_test)

print(classification_report(y_test, y_pred_mlp))

# %%
