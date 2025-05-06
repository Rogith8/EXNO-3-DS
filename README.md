## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
Developed by: ROGITH K
Register no: 212223110042
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-05-06 192445](https://github.com/user-attachments/assets/87b40d38-0d37-4ea1-97ef-ffdcb8681363)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-05-06 192459](https://github.com/user-attachments/assets/33b15563-bb63-4a5c-b20e-70cc91d64f77)
```
# Label Encoder (Orders in Alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-05-06 192514](https://github.com/user-attachments/assets/0fa0c950-0adb-431c-8ffa-abe9533947d0)
```
# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df
```
![Screenshot 2025-05-06 192533](https://github.com/user-attachments/assets/a291fc23-9d50-48f2-be3d-b8bb9663bdfb)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-05-06 192548](https://github.com/user-attachments/assets/99ccd267-d483-4287-8419-1a74bc0978b2)
```
pip install --upgrade category_encoders
```
![Screenshot 2025-05-06 192610](https://github.com/user-attachments/assets/faabf7d3-f2b5-4d47-af9f-7ee41f4a38ec)
```
from category_encoders import BinaryEncoder
df = pd.read_csv("/content/data.csv")
df
```
![Screenshot 2025-05-06 192632](https://github.com/user-attachments/assets/c8b04900-5a80-4b31-9898-625c35ade817)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```
![Screenshot 2025-05-06 192651](https://github.com/user-attachments/assets/9982ac86-1e0c-49bf-9534-73f412625f6e)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2025-05-06 192707](https://github.com/user-attachments/assets/3139678e-b8c9-46d7-b23e-22d52c76f293)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2025-05-06 192732](https://github.com/user-attachments/assets/3d23c88c-0716-4567-b382-34a8f1e407ea)
```
df.skew()
```
![Screenshot 2025-05-06 192756](https://github.com/user-attachments/assets/5a6048e3-909b-4ee9-a3ef-a33590d3e0ae)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-05-06 192813](https://github.com/user-attachments/assets/f7d0ada4-2a88-4402-bd55-f9d432dfdba7)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-05-06 192829](https://github.com/user-attachments/assets/5f8fa5d3-b1d8-4c4a-b84a-51030d0fb10e)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-05-06 192928](https://github.com/user-attachments/assets/fb8e609e-7c72-4433-b736-7fbd54ad904c)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-05-06 192941](https://github.com/user-attachments/assets/b72cecd9-12ba-4201-a7fd-be3d6da223a4)
```
# POWER TRANSFORMATION
# BOX_COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2025-05-06 193000](https://github.com/user-attachments/assets/b4b69c66-5d09-421f-8c17-67e25eef82d5)
```
df.skew()
```
![Screenshot 2025-05-06 193013](https://github.com/user-attachments/assets/a7d780a6-bf2a-41e0-a950-08554a06d2be)
```
# YEO JOHNSON
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-05-06 193024](https://github.com/user-attachments/assets/de4295c7-d499-4c58-bb63-137f0eda7687)
```
# QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-05-06 193047](https://github.com/user-attachments/assets/0efc643a-eaba-4b31-9798-7cbb86a27034)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-05-06 193104](https://github.com/user-attachments/assets/625c589a-f630-4320-bcce-5c830a2735dd)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-05-06 193114](https://github.com/user-attachments/assets/8e76bad6-e100-4107-b458-cd7680ae7cf7)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-05-06 193122](https://github.com/user-attachments/assets/c751706a-4267-4477-878d-54a24a747cda)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-05-06 193131](https://github.com/user-attachments/assets/261ad1cc-4afb-4513-b443-e019e37d8ed7)



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
