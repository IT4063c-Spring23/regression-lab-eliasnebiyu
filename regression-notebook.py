#!/usr/bin/env python
# coding: utf-8

# # Regression Lab
# 
# If you're already here, chances are that you can ignore this statement: Please make sure you follow the installation and setup guides on the assignment `README.md` file.
# 
# **Objectives**
# - Practice training Linear Regression models
# - Practice training Multiple Linear Regression models
# - Practice training Polynomial Regression models
# - Practice training Multiple Polynomial Regression models
# 
# 
# **Emojis Legend**
# - 👨🏻‍💻 - Instructions; Tells you about something specific you need to do.
# - 🦉 - Tips; Will tell you about some hints, tips and best practices
# - 📜 - Documentations; provides links to documentations
# - 🚩 - Checkpoint; marks a good spot for you to commit your code to git
# - 🕵️ - Tester; Don't modify code blocks starting with this emoji

# ## Setup
# First, let's import a few common modules, ensure `MatplotLib` plots figures inline. We also ensure that you have the correct version of Python (3.10) installed.
# 
# - **Task 👨🏻‍💻**: Keep coming back to update this cell as you need to import new packages.
# - **Task 👨🏻‍💻**: Check what's already been imported here

# In[ ]:


# Python ≥3.10 is required
import sys
assert sys.version_info >= (3, 10)

# Common imports
import numpy as np
import pandas as pd
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

# other imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# ## Education-Seniority-Income Data
# ### EDA
# The following dataset is a collection of data from a survey of 30 people. The data contains the following columns:
# - `education`: Years of education
# - `Seniority`: (months?) of work experience
# - `Income`: Income in thousand dollars
# 
# <details>
#   <summary>Data should look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/education-seniority-income-dataset.png" />
# </details>

# **Task 👨🏻‍💻**: Import the (income2.csv) dataset into a Pandas DataFrame:
# 1. name the DataFrame `income_df`
# 2. Print the first 5 rows

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: print the DataFrame's information <ins>and</ins> statistical summary
# 
# _hint:_ wrap your function calls in a `display()` function call

# In[ ]:


# FIXME
display(...)
display(...)


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Get the correlation matrix of your dataset

# In[ ]:





# **Task 👨🏻‍💻**: Plot the correlation matrix of the DataFrame as a heatmap
# <details>
#   <summary>Graph should look like this:</summary>
#   <p>this was created using `seaborn`'s heatmap function</p>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income2-heatmap.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Plot the correlation matrix of the DataFrame as a scatter matrix of charts showing the relationship between each pair of variables
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income2-scatter.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: using `Plotly` create a 3D scatter plot of the data
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income-3d.gif" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# ### Simple Linear Regression

# **Task 👨🏻‍💻**: Chart a scatter plot of `education` vs `income`
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income2-edu-income-scatterplot.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Use `sklearn` to train a Linear Regression model on the `education` and `income` columns
# - Just instantiating the model, and fitting the data is enough

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Plot the trained model (line) on top of the scatter plot
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/edu-income-linear-model.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Calculate the model's `MSE` score

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Use the `Predict` method to predict the income of a person with 20 years of education
# 
# _hint:_ it's going to be `85.82661213`

# In[ ]:





# > 🚩 : Make a git commit here

# **✨ Extra Credit Task 👨🏻‍💻**: For 6 Points:
# _HARD:_ Plot the cost function of the model. This should be a bowl shaped graph with a minimum at the bottom. The minimum is the point where the model is the most accurate.

# In[ ]:





# > 🚩 : Make a git commit here

# ### Multiple Linear Regression

# **Task 👨🏻‍💻**: Use `sklearn` to train a Linear Regression model on the `education`, `seniority` and `income` columns
# - Just instantiating the model, and fitting the data is enough

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Plot the trained model (surface) on top of the scatter plot
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/income-multiple-linear-regression.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **✨ Extra Credit Task 👨🏻‍💻**: For 3 Points: Plot the trained model (surface) on top of the scatterplot using `Plotly`'s interactive 3d diagrams

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Use the `Predict` method to predict the income of a person with `15` years of education, and seniority score of `75`
# 
# 
# _Hint:_ it should be `51.31186086`

# In[ ]:





# > 🚩 : Make a git commit here

# **Task ❓👨🏻‍💻**: (open ended) Reflect on the linear and polynomial regression models you've built. Which model do YOU think is more appropriate? and why?
# 
# _Hint:_ you can use the `MSE` score to help you decide

# In[ ]:


#👨🏻‍💻 if you're coding something to answer the question, do it here in this cell.

# 👨🏻‍💻 provide your answer here:

# > 🚩 : Make a git commit here

# ### Polynomial Regression

# **Task 👨🏻‍💻**: Use `sklearn` to train a Polynomial Regression model (of the second degree) on the `education` and `income` columns
# - Just instantiating the model, and fitting the data is enough
# - You'll need to use the `PolynomialFeatures` class to transform the data into a polynomial form

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Plot the trained model (curve) on top of the scatter plot
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/edu-income-polynomial-model.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Use the `Predict` method to predict the income of a person with `11` years of education
# 
# 
# _Hint:_ 
# - it should be `25.48739205`
# - You will need to create a numpy array of the input data, and reshape it. `np.array([11]).reshape(1, -1)`
# - You will need to `fit-transform` the input data before you can `predict` it

# In[ ]:





# > 🚩 : Make a git commit here

# ### Multiple Polynomial Regression

# **Task 👨🏻‍💻**: Use `sklearn` to train a Multiple Polynomial Regression model (of the third degree) on the `education`, `seniority` and `income` columns
# 
# _Hint:_
# - Just instantiating the model, and fitting the data is enough
# - You'll need to use the `PolynomialFeatures` class to transform the data into a polynomial form

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻**: Plot the trained model (curved surface) on top of the scatter plot
# <details>
#   <summary>Graph would look like this:</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/regression-assignment/3rd-dec-income-surface.png" />
# </details>

# In[ ]:





# > 🚩 : Make a git commit here

# **✨ Extra Credit Task 👨🏻‍💻**: For 3 Points: Plot the trained model (curved surface) on top of the scatterplot using `Plotly`'s interactive 3d diagrams

# In[ ]:





# > 🚩 : Make a git commit here

# ## Wrap up
# Remember to update the self reflection and self evaluations on the `README` file.

# #### 🦉: MAKE SURE:
# To run the following cell: The following command converts this Jupyter notebook to a Python script. This allows me to provide feedback on your code.

# In[ ]:


get_ipython().system('jupyter nbconvert --to python regression-notebook.ipynb')


# > 🚩 : Make a git commit here
