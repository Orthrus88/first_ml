import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import csv dataset as df or dataframe
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# Data Prep on csv dataset

# define output this is what the ML will guess from the dataset
y = df['logS']

# define inputs as x minus the y of logS. Set axis to 1 so it remains a column to work with later. axis 0 is for row.
x = df.drop('logS', axis=1)


# Split dataset for training and testing using sklearn
# define x train / test and y train / test equal to train_test_split function
# train test split will house the value of X, Y, test size of 20% of the data, and a random state of 100 that will determin the shuffling of the data.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Build the model
# linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# making predictions
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Eval Model Performance
# Compare the training model answers to the training model predicitons
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# Compare the test model answers to the training model predictions
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print(lr_train_mse)