#!/usr/bin/env python
# coding: utf-8

# ███████████  ████████████      ████████      ████████
# ███████████  ███████████████   █████████    █████████
#    █████        ████   █████     ████████  ████████
#    █████        ███████████      ████  ███ ███ ████
#    █████        ███████████      ████  ███████ ████
#    █████        ████   █████     ████   █████  ████
# ███████████  ███████████████   ██████    ███   ██████
# ███████████  ████████████      ██████     █    ██████

# #  IBM 'Perfect Employee' Project & Employee Churn 🌐
# ---
# ## What Makes The Best Employee ? || Why Do Workers Quit?
# ### Objective: Investigate the IBM Employee Data Set to gain sufficient understanding of what attributes to the best employee and in turn utilize this analysis to create actionable recommendations for IBMs hiring process procedure.
# ---
# 
# Employees are what drives company growth. An organization's performance is heavily based on the quality of the employees. Without the right team of employees , the company has no future. 
# 
# ------
# ##  Business questions to keep in mind: 🧠
# ---
# > 1. What factors are contributing more to employee daily rate?
# > 2. What contributes to employee attrition?
# > 3. Will this data help employers in the future of hiring the right employee based on the data ?
# 

# In[54]:


get_ipython().system('pip install -q hvplot')


# In[55]:


import hvplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import hvplot.pandas

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 80)


# In[56]:


#Setting the default style for plots
plt.style.use('ggplot')

from matplotlib.pyplot import figure
plt.rcParams['figure.figsize'] = (12,8)

get_ipython().run_line_magic('matplotlib', 'inline')


# ##  Deep Diving into the Data: 🔍
# 
# > 1. Let see what the data entails and get a birds eye view on the complete overview.
# 

# In[57]:


data = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')

data.head()


# In[58]:


data2.head()


# In[59]:


data.describe()


# In[60]:


for column in data.columns:
    print(f"{column}: Number of unique values {data[column].nunique()}")
    print("          ")


# In[61]:


data.info()




# In[62]:


for i in data:
    print("number of NaN values for the column " + i + " =" , data[i].isnull().sum())
    print("          ")


# In[63]:


for i in data:
    print("number of NaN values for the column " + i + " =" , data[i].isna().sum())
    print("          ")


# In[64]:


cont_col = []
for column in data.columns:
    if data[column].dtypes != object and data[column].nunique() > 30:
        print(f"{column} : Minimum: {data[column].min()}, Maximum: {data[column].max()}")
        cont_col.append(column)
        print("          ")


# # Question 1: Who Are The Top 15 Highest Paid Employees In The Company ?
# > Now we can start asking the real questions to find out what makes the best employee.

# In[191]:


#data.sort_values(by=['EmployeeNumber', 'DailyRate'])
data.sort_values(by=['MonthlyIncome'] , ascending=False).head(15)



# # Question 2: Do Employees Education Background Have An Impact On Their Monthly Rate ?

# In[172]:


plt.hist(data['EducationField'])
plt.xlabel('EducationField')
plt.ylabel('MonthlyRate')



#  # Question 3: Does The Longer An Employee Work At The Company Have A Correlation To Their Pay ?

# In[173]:


plt.hist(data['YearsAtCompany'])
plt.xlabel('YearsAtCompany')
plt.ylabel('MonthlyRate')


# # Question 4: Which Departments Make The Most Money ?

# In[175]:


g1 = data.groupby('Department')['MonthlyRate'].count()
explode = (0, 0.1, 0)
g1.plot(kind = 'pie' , shadow = True , explode=explode , autopct='%1.1f%%')


# In[171]:


g1 = data.groupby('Gender')['EmployeeNumber'].count()
explode = (0, 0.1,)
g1.plot(kind = 'pie' , shadow = True , explode=explode , autopct='%1.1f%%')


#  # Question 5: The Last Year Since An Employee Got Promoted Increase Their Chances Of Pay ?

# In[176]:


plt.hist(data['YearsSinceLastPromotion'])
plt.xlabel('YearsSinceLastPromotion')
plt.ylabel('MonthlyRate')


#  # Question 6: The Longer You've worked with the current manager have a benefit ?

# In[177]:


plt.hist(data['YearsWithCurrManager'])
plt.xlabel('YearsWithCurrManager')
plt.ylabel('MonthlyRate')


#  # Question 7: Could Potentially Switching Roles , Give You An Edge ?

# In[178]:


plt.hist(data['YearsInCurrentRole'])
plt.xlabel('YearsInCurrentRole')
plt.ylabel('MonthlyRate')


#  # Question 8: Do Employees Who Travel More Do It For A Reason ?

# In[180]:


plt.hist(data['BusinessTravel'])
plt.xlabel('BusinessTravel')
plt.ylabel('MonthlyRate')


#  # Question 9: Is Business Travel Maybe A Driving Factor In Poor Work Life Balance

# In[233]:


plt.hist(data['BusinessTravel'])
plt.xlabel('BusinessTravel')
plt.ylabel('WorkLifeBalance')


#  # Question 10: If Your Loyal To The Company For Years , Does It Pay Off ?

# In[181]:


plt.hist(data['YearsAtCompany'])
plt.xlabel('YearsAtCompany')
plt.ylabel('MonthlyRate')


#  # Question 11: Workers Who Train During The Year Have An Advantage ?

# In[185]:


plt.hist(data['TrainingTimesLastYear'])
plt.xlabel('TrainingTimesLastYear')
plt.ylabel('MonthlyRate')


#  # Question 12: Job Satisfaction X Monthly Rate ?

# In[197]:


plt.figure(figsize=(10,6))
plt.hist(data['JobSatisfaction'])
plt.xlabel('JobSatisfaction')
plt.ylabel('MonthlyRate')


#  # Question 13: What Job Roles Are The Best ?

# In[224]:


plt.figure(figsize=(19,10))
plt.hist(data['JobRole'])
plt.xlabel('JobRole')
plt.ylabel('MonthlyRate')


# In[223]:


plt.figure(figsize=(19,10))
plt.hist(data['JobRole'])
plt.xlabel('JobRole')
plt.ylabel('WorkLifeBalance')


#  # Question 14: Does coming with more experience from other companies make you a better candidate ?

# In[216]:


plt.figure(figsize=(10,6))
plt.hist(data['NumCompaniesWorked'])
plt.xlabel('NumCompaniesWorked')
plt.ylabel('MonthlyRate')


# In[196]:


data.hvplot.hist(y='JobLevel', by='Attrition', subplots=False, width=600, height=300)


# In[207]:


data.hvplot.hist(y='Age', by='Attrition', subplots=False, width=600, height=300)


# In[209]:


data.hvplot.hist(y='MonthlyRate', by='Attrition', subplots=False, width=600, height=300)


# In[210]:


data.hvplot.hist(y='YearsAtCompany', by='Attrition', subplots=False, width=600, height=300)


# In[211]:


data.hvplot.hist(y='YearsInCurrentRole', by='Attrition', subplots=False, width=600, height=300)


# In[212]:


data.hvplot.hist(y='WorkLifeBalance', by='Attrition', subplots=False, width=600, height=300)


# In[214]:


data.hvplot.hist(y='DistanceFromHome', by='Attrition', subplots=False, width=600, height=300)


# In[215]:


data.hvplot.hist(y='NumCompaniesWorked', by='Attrition', subplots=False, width=600, height=300)


# 
# ***
# 
# ## 💻  **Conclusions:**
# 
# ***
# - Workers coming from a `EducationField` of Life Sciences and Medical are the top paid workers.
# 
# - Sales Executives and Research Scientist are the higest paid `JobRoles` as well.
# 
# - The workers in `Laboratory Technician`, `Sales Representative`, and `Human Resources` are more likely to quit the workers in other positions.
# 
# - The more recent you got promoted the higher chance youll get an increase in `MonthlyRate`
# 
# - The workers with low `JobLevel`, `MonthlyIncome`, `YearAtCompany`, and `TotalWorkingYears` are more likely to quit there jobs.
# 
# - Employees who stay 2.5 Years with the current manager have the best return on their time. 
# 
# - `BusinessTravel` : The workers who travel alot are more likely to quit then other employees.
# 
# - `Department` : The worker in `Research & Development` are more likely to stay then the workers on other departement.
# 
# - `EducationField` : The workers with `Human Resources` and `Technical Degree` are more likely to quit then employees from other fields of educations.
# 
# - Employees with the higher `JobSatisfaction` have higer rates as well.
# 
# - `Gender` : The `Male` are more likely to quit.
# 
# - `OverTime` : The workers who work more hours are likely to quit then others.
# 
# - `MaritalStatus` : The workers who have `Single` marital status are more likely to quit the `Married`, and `Divorced`.
# 
# 
# *** 
# 

# In[ ]:




