#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
"""
Modulo 3 - Decision model discovery
"""

# Commented out IPython magic to ensure Python compatibility.
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from decisiondiscovery.views import plot_decision_tree
from sklearn.ensemble import RandomForestClassifier
# %matplotlib inline

param_preprocessed_log_path = sys.argv[1] if len(sys.argv) > 1 else "media/preprocessed_dataset.csv"
param_path = sys.argv[2] if len(sys.argv) > 2 else "media/"
autogeneration = sys.argv[3] if len(sys.argv) > 3 else 'normal'

df = pd.read_csv(param_preprocessed_log_path,index_col=0, sep=',')
# df.head()
# df.info()

one_hot_cols = []
text_cols = []
for c in df.columns:
  if "NameApp" in c:
    one_hot_cols.append(c)
  elif "TextInput" in c:
    text_cols.append(c)

# print("\n\nColumns to drop: ")
# print(one_hot_cols)
# print(text_cols)

# for c in one_hot_cols:
#  df[c] = df[c].map(dict(zip(['Firefox','CRM'],[0,1])))
df = pd.get_dummies(df, columns=one_hot_cols)
df = df.drop(text_cols, axis=1)
# df.head()
# sns.countplot(df['Variant'])

df = df.fillna(0.)

# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)

# np.any(np.isnan(df))
# np.isnan(df)
# np.all(np.isfinite(df))

from sklearn.model_selection import train_test_split
X = df.drop('Variant',axis=1)
y = df[['Variant']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   
# clf_model = RandomForestClassifier(n_estimators=100)   
clf_model.fit(X_train,y_train)

y_predict = clf_model.predict(X_test)

# # print("\nTest dataset: ")
# # print(X_test)
# # print("\nCorrect labels: ")
# # print(y_test)
# # print("\nTest dataset predictions: ")
# # print(y_predict)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# print("\n\nAccuracy_score")
# print(accuracy_score(y_test,y_predict))

target = list(df['Variant'].unique())
feature_names = list(X.columns)

target_casted = [str(t) for t in target]

# estimator = clf_model.estimators_[5]

# from sklearn.tree import export_graphviz

# export_graphviz(estimator, out_file='tree.dot', 
#                 feature_names = feature_names,
#                 class_names = target_casted,
#                 rounded = True, proportion = False, 
#                 precision = 2, filled = True)

# # Convert to png using system command (requires Graphviz)
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# # Display in jupyter notebook
# from IPython.display import Image
# Image(filename = 'tree.png')

from sklearn.tree import export_text
text_representation = export_text(clf_model, feature_names=feature_names)
# print("\n\nDecision Tree Text Representation")
# print(text_representation)

with open(param_path + "decision_tree.log", "w") as fout:
    fout.write(text_representation)

type(target_casted[0])

if not autogeneration=='autogeneration':
  img = plot_decision_tree(param_path + "decision_tree", clf_model,feature_names,target_casted)
  plt.imshow(img)
  plt.show()