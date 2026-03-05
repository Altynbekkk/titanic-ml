import pandas as pd
import numpy as np
import os

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# -------------------------------
# 1️⃣ Загрузка данных
# -------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Сохраняем PassengerId для submission
test_ids = test["PassengerId"]

# -------------------------------
# 2️⃣ Feature Engineering
# -------------------------------
rare_titles = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]

for dataset in [train, test]:
    # Sex → 0/1
    dataset["Sex"] = dataset["Sex"].map({"male":0,"female":1})
    
    # Family
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset["IsAlone"] = (dataset["FamilySize"] == 1).astype(int)
    
    # Embarked
    dataset["Embarked"] = dataset["Embarked"].fillna("S")
    dataset["Embarked"] = dataset["Embarked"].map({"S":0,"C":1,"Q":2})
    
    # Title
    dataset["Title"] = dataset["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    dataset["Title"] = dataset["Title"].replace(rare_titles, "Rare")
    dataset["Title"] = dataset["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
    dataset["Title"] = dataset["Title"].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Rare":4})
    
    # Fare
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    dataset["Fare"] = np.log1p(dataset["Fare"])  # сглаживаем

    # Age → теперь можно заполнять по Title
    dataset["Age"] = dataset.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

# -------------------------------
# 3️⃣ Features и target
# -------------------------------
features = ["Pclass","Sex","Age","Fare","FamilySize","IsAlone","Embarked","Title"]
X_train = train[features]
y_train = train["Survived"]
X_test = test[features]

# -------------------------------
# 4️⃣ XGBoost модель
# -------------------------------
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

# -------------------------------
# 5️⃣ Cross-validation
# -------------------------------
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("CV Accuracy: %.4f ± %.4f" % (scores.mean(), scores.std()))

# -------------------------------
# 6️⃣ Обучаем на всех данных
# -------------------------------
model.fit(X_train, y_train)
pred = model.predict(X_test)

# -------------------------------
# 7️⃣ Submission
# -------------------------------
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": pred
})
submission.to_csv("submission.csv", index=False)