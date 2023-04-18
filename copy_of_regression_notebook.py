# -*- coding: utf-8 -*-
"""Copy of regression-notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BLRN35GRqxcJa-nDsB1EVPSwwPZI2cwb

# Regression Lab

## Introduction
Regression analysis is an important statistical technique used to model the relationship between a dependent variable and one or more independent variables. It is commonly used in various fields, such as finance, economics, engineering, and social sciences, to make predictions and understand the underlying patterns in data.

## Learning Objectives
- Practice training Linear Regression models
- Practice training Multiple Linear Regression models
- Practice training Polynomial Regression models
- Practice training Multiple Polynomial Regression models


**Emojis Legend**
- 👨🏻‍💻 - Instructions; Tells you about something specific you need to do.
- 🦉 - Tips; Will tell you about some hints, tips and best practices
- 📜 - Documentations; provides links to documentations
- 🚩 - Checkpoint; marks a good spot for you to commit your code to git
- 🕵️ - Tester; Don't modify code blocks starting with this emoji

## Setup
* Install this lab's dependencies by running the following command in your terminal: `pipenv install`
* Make sure you switch to the correct environment by choosing the correct kernel in the top right corner of the notebook.

### Package Imports
We will keep coming back to this cell to add "import" statements, and configure libraries as we need

- **Task 👨🏻‍💻**: Keep coming back to update this cell as you need to import new packages.
- **Task 👨🏻‍💻**: Check what's already been imported here
"""

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
import pandas as pd

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

# other imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

"""## Education-Seniority-Income Data
### EDA
The following dataset is a collection of data from a survey of 30 people. The data contains the following columns:
- `education`: Years of education
- `Seniority`: (months?) of work experience
- `Income`: Income in thousand dollars

<details>
  <summary>Data should look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/education-seniority-income-dataset.png" />
</details>

**Task 👨🏻‍💻**: Import the (income2.csv) dataset into a Pandas DataFrame:
1. name the DataFrame `income_df`
2. Print the first 5 rows
"""

income_df = pd.read_csv('Income2.csv')

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: print the DataFrame's information <ins>and</ins> statistical summary

_hint:_ wrap your function calls in a `display()` function call so you can put them both in the cell
"""

# print information about the DataFrame
print(income_df.info())

# print statistical summary of the DataFrame
print(income_df.describe())

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Get the correlation matrix of your dataset
"""

income_df.corr()

"""**Task 👨🏻‍💻**: Plot the correlation matrix of the DataFrame as a heatmap
<details>
  <summary>Graph should look like this:</summary>
  <p>this was created using `seaborn`'s heatmap function</p>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income2-heatmap.png" />
</details>
"""

#create a correlation matrix
corr_matrix = income_df.corr()

# plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Plot the correlation matrix of the DataFrame as a scatter matrix of charts showing the relationship between each pair of variables
<details>
  <summary>Graph would look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income2-scatter.png" />
</details>
"""

# plot a scatter matrix of the DataFrame
pd.plotting.scatter_matrix(income_df, diagonal='hist')
plt.suptitle('Scatter Matrix of Variables')
plt.show()

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: using `Plotly` create a 3D scatter plot of the data
<details>
  <summary>Graph would look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income-3d.gif" />
</details>
"""

# create a 3D scatter plot of the DataFrame
fig = go.Figure(data=[go.Scatter3d(
    x=income_df['education'],
    y=income_df['seniority'],
    z=income_df['income'],
    mode='markers',
    marker=dict(
        size=10,
        color='blue',
        opacity=0.8
    )
)])

# set plot title and axis labels
fig.update_layout(title='3D Scatter Plot', scene=dict(
                    xaxis_title='Education',
                    yaxis_title='Seniority',
                    zaxis_title='Income'))

fig.show()

"""> 🚩 : Make a git commit here

### Simple Linear Regression

**Task 👨🏻‍💻**: Chart a scatter plot of `education` vs `income`
<details>
  <summary>Graph would look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income2-edu-income-scatterplot.png" />
</details>
"""

# plot a scatter plot of Education vs. Income
plt.scatter(income_df['education'], income_df['income'])
plt.xlabel('Education')
plt.ylabel('Income')
plt.title('Education vs. Income Scatter Plot')
plt.show()

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Use `sklearn` to train a Linear Regression model on the `education` and `income` columns
- Just instantiating the model, and fitting the data is enough
"""

# instantiate a linear regression model
model = LinearRegression()

# fit the model to the data
model.fit(income_df[['education']], income_df['income'])

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Plot the trained model (line) on top of the scatter plot
<details>
  <summary>Graph would look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/edu-income-linear-model.png" />
</details>
"""

# plot the scatter plot of Income vs. Education
plt.scatter(income_df['education'], income_df['income'], color='blue')
plt.xlabel('Education')
plt.ylabel('Income')
plt.title('Income vs. Education Scatter Plot')

# plot the trained model as a line
plt.plot(income_df['education'], model.predict(income_df[['education']]), color='red')
plt.show()

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Use the `Predict` method to predict the income of a person with 20 years of education

_hint:_ it's going to be `85.82661213`
"""

# predict the income of a person with 20 years of education
edu_20 = [[20]]
predicted_income = model.predict(edu_20)
print(predicted_income)

"""> 🚩 : Make a git commit here

### Multiple Linear Regression

**Task 👨🏻‍💻**: Use `sklearn` to train a Linear Regression model on the `education`, `seniority` and `income` columns
- Just instantiating the model, and fitting the data is enough
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# extract the input features and output variable
X = income_df[['education', 'seniority']]
y = income_df['income']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate a linear regression model
model = LinearRegression()

# fit the model to the training data
model.fit(X_train, y_train)

# evaluate the model on the testing data
score = model.score(X_test, y_test)

print(f"R^2 score of the model: {score}")

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Plot the trained model (surface) on top of the scatter plot
* You can choose to use `Plotly` or `matplotlib` for this
<details>
  <summary>Graph would look something like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income-multiple-linear-regression.png" />
</details>
"""

import plotly.express as px
# create a grid of points to evaluate the model
x = np.linspace(income_df['education'].min(), income_df['education'].max(), 50)
y = np.linspace(income_df['seniority'].min(), income_df['seniority'].max(), 50)
X, Y = np.meshgrid(x, y)
Z = model.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# create a 3D scatter plot of the data
fig = px.scatter_3d(income_df, x='education', y='seniority', z='income')

# add the model surface to the plot
fig.add_trace(go.Surface(x=x, y=y, z=Z))

# show the plot
fig.show()

"""> 🚩 : Make a git commit here

**✨ Extra Credit Task 👨🏻‍💻**: For 3 Points: Plot the trained model (surface) on top of the scatterplot using the other package
* If you used `Plotly`, use `matplotlib` and vice versa
"""

# create a meshgrid of points to evaluate the model
x = np.linspace(income_df['education'].min(), income_df['education'].max(), 50)
y = np.linspace(income_df['seniority'].min(), income_df['seniority'].max(), 50)
X, Y = np.meshgrid(x, y)
Z = model.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# create a 3D scatter plot of the data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(income_df['education'], income_df['seniority'], income_df['income'], s=100, c='b', marker='o')
ax.set_xlabel('Education')
ax.set_ylabel('Seniority')
ax.set_zlabel('Income')

# plot the model surface on top of the scatter plot
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='jet')

plt.show()

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Use the `Predict` method to predict the income of a person with `15` years of education, and seniority score of `75`


_Hint:_ it should be `51.31186086`
"""

prediction = model.predict([[15, 75]])
print("Predicted income:", prediction[0])

"""> 🚩 : Make a git commit here

### Polynomial Regression

**Task 👨🏻‍💻**: Use `sklearn` to train a Polynomial Regression model (of the second degree) on the `education` and `income` columns
- Just instantiating the model, and fitting the data is enough
- You'll need to use the `PolynomialFeatures` class to transform the data into a polynomial form
"""

# Extract the input and target variables
X = income_df[['education']]
y = income_df['income']

# Transform the input data into polynomial form
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Instantiate and fit the model
model = LinearRegression()
model.fit(X_poly, y)

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Plot the trained model (curve) on top of the scatter plot
<details>
  <summary>Graph would look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/edu-income-polynomial-model.png" />
</details>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Extract the input and target variables
X = income_df[['education']]
y = income_df['income']

# Transform the input data into polynomial form
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the model to the transformed data
model = LinearRegression()
model.fit(X_poly, y)

# Use the model to make predictions
y_pred = model.predict(X_poly)

# Plot the scatter plot and the fitted curve
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel('Education')
plt.ylabel('Income')
plt.show()

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Use the `Predict` method to predict the income of a person with `11` years of education


_Hint:_ 
- it should be `25.48739205`
- You will need to create a numpy array of the input data, and reshape it. `np.array([11]).reshape(1, -1)`
- You will need to `fit-transform` the input data before you can `predict` it
"""

# transform the input data into a polynomial form
X_test_poly = poly.transform([[11]])

# use the trained model to make a prediction
y_pred = model2.predict(X_test_poly)

# print the predicted income
print('Predicted income:', y_pred[0])

"""> 🚩 : Make a git commit here

### Multiple Polynomial Regression

**Task 👨🏻‍💻**: Use `sklearn` to train a Multiple Polynomial Regression model (of the third degree) on the `education`, `seniority` and `income` columns

_Hint:_
- Just instantiating the model, and fitting the data is enough
- You'll need to use the `PolynomialFeatures` class to transform the data into a polynomial form
"""

# Transform data into polynomial form
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Instantiate and fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Use the model to make predictions
y_pred = model.predict(X_poly)

"""> 🚩 : Make a git commit here

**Task 👨🏻‍💻**: Plot the trained model (curved surface) on top of the scatter plot
* You can choose to use `Plotly` or `matplotlib` for this
<details>
  <summary>Graph would look like this:</summary>
  <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/3rd-dec-income-surface.png" />
</details>
"""

# Extract the input and target variables
X = income_df[['education', 'seniority', 'income']]
y = income_df['income']

# Transform the input data into polynomial form
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit the model to the transformed data
model = LinearRegression()
model.fit(X_poly, y)

# Use the model to make predictions
# We need to create a new observation with values for each feature
new_observation = np.array([[15, 75, 0]])  # 15 years of education, 75 seniority score, and unknown income
new_observation_poly = poly.transform(new_observation)
y_pred = model.predict(new_observation_poly)

print(f"The predicted income for a person with 15 years of education, 75 seniority score, and unknown income is ${y_pred[0]:.2f}")

"""> 🚩 : Make a git commit here

**✨ Extra Credit Task 👨🏻‍💻**: For 3 Points: Plot the trained model (curved surface) on top of the scatterplot using the other package
* If you used `Plotly`, use `matplotlib` and vice versa
"""



"""> 🚩 : Make a git commit here

## Wrap up
### 📝 Reflection
- What did you learn from this assignment?
- What was the most challenging part of this assignment?
- What would you do differently next time?

### Citations
Cite any resources you used to complete this assignment

This includes: 
- Individuals other than the instructor
- Websites
- Videos
- AI assistants such as GitHub Copilot or ChatGPT

#### 🦉: MAKE SURE YOU RUN THE FOLLOWING CELL BEFORE SUBMITTING
The following command converts this Jupyter notebook to a Python script. This allows me to provide feedback on your code.
"""

!jupyter nbconvert --to python regression-notebook.ipynb

"""> 🚩 : Make a git commit here"""