#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
"""
Modulo 3 - Decision model discovery
"""

# Commented out IPython magic to ensure Python compatibility.
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

param_preprocessed_log_path = sys.argv[1]

df = pd.read_csv(param_preprocessed_log_path,index_col=0, sep=';')
# df.head()
# df.info()

one_hot_cols = []
text_cols = []
for c in df.columns:
  if "NameApp" in c:
    one_hot_cols.append(c)
  elif "TextInput" in c:
    text_cols.append(c)

print("\n\nColumns to drop: ")
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



print("\n\nConjunto de prueba: ")
print(X_test)
print("\n\nEtiquetas reales: ")
print(y_test)
print("\n\nPredicciones sobre el conjunto de pruebas: ")
print(y_predict)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print("\n\nAccuracy_score")
print(accuracy_score(y_test,y_predict))

target = list(df['Variant'].unique())
feature_names = list(X.columns)

from sklearn.tree import export_text
text_representation = export_text(clf_model, feature_names=feature_names)
print("\n\nDecision Tree Text Representation")
print(text_representation)

with open("media/decistion_tree.log", "w") as fout:
    fout.write(text_representation)

target_casted = [str(t) for t in target]
type(target_casted[0])

# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf_model, 
#                    feature_names=feature_names,  
#                    class_names=target_casted,
#                    filled=True)

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf_model,
                                out_file=None, 
                      feature_names=feature_names,  
                      class_names=target_casted,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  

# graph

graph.save('media/graph1.jpg')