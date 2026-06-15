#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir(r"C:\Users\sathu\Projects")
print(os.getcwd())


# - Step-1: Load the data set(already stored in the files as CSV)
# - Step-2: Inspect the dataset
# - Step-3: Separate Input and Output
# - Step-4: Visualize the data
# - Step-5: Feature Scaling
# - Step-6: Split the dataset
# - Step-7: Create a model
# - Step-8: Train the model
# - Step-9: Make PRedictions
# - Step-10: Compare Actual Vs Predicted Values
# - Step-11: Plot the Regression Line
# - Step-12: Try Custom Inputs

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv('Salary_dataset.csv')
data.head()


# In[4]:


data.info()


# In[5]:


x = data[['YearsExperience']]
y = data[['Salary']]


# In[6]:


plt.scatter(x,y)
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.title('Salary Vs. Experience')
plt.show()


# In[7]:


x_raw = data['YearsExperience'].values
y_raw = data['Salary'].values

x_mean, x_std = np.mean(x_raw), np.std(x_raw)
y_mean, y_std = np.mean(y_raw), np.std(y_raw)

X = (x_raw - x_mean) / x_std
Y = (y_raw - y_mean) / y_std


# In[8]:


w = 1.0
b = 1.0
def compute_cost(X,Y,w,b):
    m = len(X)
    predictions = w * X + b
    cost = (1/(2*m)) * np.sum((predictions-Y) ** 2)
    return cost
def gradient_descent(X,Y,w,b,alpha_,iterations):
    m = len(X)
    cost_history = []
    for i in range(iterations):
        f_x = w * X + b #predictions
        #calculate gradients
        dj_dw = (1/m)*np.sum((f_x - Y)*X)
        dj_db = (1/m)*np.sum(f_x - Y)
        #update parameters
        w = w - alpha_*dj_dw
        b = b - alpha_*dj_db
        # track cost
        cost_history.append(compute_cost(X,Y,w,b))
    return w,b,cost_history


# In[9]:


alpha_ = 0.01
iterations = 1000
w_opt,b_opt,cost_history = gradient_descent(X,Y,w,b,alpha_,iterations)
print(f"Optimized w: {w_opt:.4f}")
print(f"Optimized b: {b_opt:.4f}")
print(f"Cost History :{cost_history[-1]:.4f}")


# In[10]:


plt.figure(figsize = (8,4))
plt.plot(cost_history,color = 'purple')
plt.title("Cost History over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()


# In[11]:


#Make Predictions
def predict_salary(years_exp):
    X_scaled = (years_exp - x_mean) / x_std #Scale the input

    y_pred_scaled = w_opt * X_scaled + b_opt #Predict

    y_pred = (y_pred_scaled * y_std) + y_mean
    return y_pred
# Example Prediction 
n = 10
print(f"Predicted salary for {n} Years experience: ${predict_salary(n):,.2f}")


# In[12]:


#plot regression line

#Generate points for the regression line across th experience range
X_line_raw = np.linspace(x_raw.min(),x_raw.max(),100)
y_line_raw = predict_salary(X_line_raw)
#plotting
plt.figure(figsize = (8,5))
plt.scatter(x_raw,y_raw,c = 'blue',label = 'Actual Data')
plt.plot(X_line_raw,y_line_raw,c = 'red',linewidth = 2,label = 'Regression Line')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.title('Salary Vs. Experience (Gradient Descent)')
plt.legend()
plt.show()


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


model = LinearRegression()


# In[16]:


X = data[['YearsExperience']]
Y = data['Salary']
model.fit(X,Y)


# In[21]:


model.coef_


# In[19]:


model.intercept_


# In[23]:


y_pred = model.predict(X)
plt.figure(figsize = (8,5))
plt.scatter(X,Y,color = 'blue',label = 'Actual Data')
plt.plot(X,y_pred,color = 'red',linewidth = 2,label = 'Regression Line')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.title('Experience Vs. Salary')
plt.grid(True)
plt.legend()
plt.show()
