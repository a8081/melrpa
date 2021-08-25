# -*- coding: utf-8 -*-
"""
Módulo 3. Decision model discovery
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv('preprocessed-dataset2.csv',index_col=0, sep=',')

# df.head()
# df.info()

one_hot_cols = []
text_cols = []
for c in df.columns:
  if "NameApp" in c:
    one_hot_cols.append(c)
  elif "TextInput" in c:
    text_cols.append(c)
print("\n\nColumns to drop:")
print(one_hot_cols)
print(text_cols)

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
clf_model.fit(X_train,y_train)

y_predict = clf_model.predict(X_test)

print(X_test)

print("\n\nConjunto de prueba: ")
print(y_test)
print("\n\nPredicciones sobre el conjunto de pruebas: ")
print(y_predict)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,y_predict)

target = list(df['Variant'].unique())
feature_names = list(X.columns)

from sklearn.tree import export_text
t_representation = export_text(clf_model, feature_names=feature_names)
print(t_representation)

# from sklearn import tree
# import graphviz
# dot_data = tree.export_graphviz(clf_model,
#                                 out_file=None, 
#                       feature_names=feature_names,  
#                       class_names=target,  
#                       filled=True, rounded=True,  
#                       special_characters=True)  
# graph = graphviz.Source(dot_data)  

# print(graph)

# graph.save('graph1.jpg')