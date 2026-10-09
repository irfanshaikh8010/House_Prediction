import sklearn
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

california_df = fetch_california_housing(as_frame=True).frame

X = california_df.drop('MedHouseVal', axis=1)
Y = california_df['MedHouseVal']

columns = X.columns.tolist()

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=1234
)

lr = LinearRegression()

lr.fit(X_train, Y_train)

d = {
    'model': lr,
    'columns': columns
}

joblib.dump(d, 'california3.joblib')

print('Model saved successfully')
print('columns:', columns)