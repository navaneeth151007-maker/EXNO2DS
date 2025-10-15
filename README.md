# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"D:\titanic_dataset.csv")
print(df.head())
```
<img width="797" height="506" alt="image" src="https://github.com/user-attachments/assets/64417779-f7d6-4764-9205-10b7049af365" />
```
print(df.info())

```
<img width="611" height="488" alt="image" src="https://github.com/user-attachments/assets/5f7a5bb9-7c4d-4c77-8e80-ced8595b19bd" />
```
print(df.describe(include='all'))
```
<img width="816" height="617" alt="image" src="https://github.com/user-attachments/assets/b175d09a-8bc5-46fc-bba7-b272165b138b" />
```
print(df.isnull().sum())
print(round(df.isnull().mean() * 100, 2))
print(df.duplicated().sum())
```
<img width="465" height="700" alt="image" src="https://github.com/user-attachments/assets/4b4e684e-5a3d-4108-bcf2-8f530be80dbb" />
```
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)
print(df)
```
<img width="832" height="726" alt="image" src="https://github.com/user-attachments/assets/9457edb5-b434-4634-ac25-5db9f1ad5785" />
```
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.show()
```
<Figure size 600x300 with 1 Axes><img width="531" height="316" alt="image" src="https://github.com/user-attachments/assets/c2970f28-6687-4fd9-8f4b-cf171eebfb57" />
<Figure size 600x300 with 1 Axes><img width="540" height="316" alt="image" src="https://github.com/user-attachments/assets/6d9d4adf-a5be-4d89-8c1e-e011b6b57a1a" />
<Figure size 600x300 with 1 Axes><img width="540" height="316" alt="image" src="https://github.com/user-attachments/assets/4c804f80-7f03-4bfb-a53f-b203a2d9563c" />
<Figure size 600x300 with 1 Axes><img width="540" height="316" alt="image" src="https://github.com/user-attachments/assets/6e5cb94e-583a-4972-933c-be4ddd5070a0" />
<Figure size 600x300 with 1 Axes><img width="540" height="316" alt="image" src="https://github.com/user-attachments/assets/a093c146-9b06-478a-9084-71caebe248a1" />
<Figure size 600x300 with 1 Axes><img width="540" height="316" alt="image" src="https://github.com/user-attachments/assets/1354e1d5-8050-480e-ba93-6fc93765e7d2" />
<Figure size 600x300 with 1 Axes><img width="540" height="316" alt="image" src="https://github.com/user-attachments/assets/21c65939-0c25-416b-bf09-1f66849b82bf" />
      
```
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(6,3))
    sns.countplot(y=col, data=df)
    plt.title(f"Count of {col}")
    plt.show()
```
<Figure size 600x300 with 1 Axes><img width="1081" height="316" alt="image" src="https://github.com/user-attachments/assets/5174d403-33a0-46f9-9ed4-4e02df8ec45b" />
<Figure size 600x300 with 1 Axes><img width="570" height="316" alt="image" src="https://github.com/user-attachments/assets/37ea74e3-f13f-41a5-bad1-740092194d86" />
<Figure size 600x300 with 1 Axes><img width="664" height="316" alt="image" src="https://github.com/user-attachments/assets/c9d102f2-f2f1-4994-821e-24fe85e6f5be" />
<Figure size 600x300 with 1 Axes><img width="636" height="316" alt="image" src="https://github.com/user-attachments/assets/c3b4b56f-1081-4960-9e76-f43329e00f1a" />
<Figure size 600x300 with 1 Axes><img width="525" height="316" alt="image" src="https://github.com/user-attachments/assets/f710f926-4536-400d-a979-ec755b1d98fd" />

```
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
```
<Figure size 600x400 with 1 Axes><img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/472df1e5-d5a3-48ed-97a9-9a9c4f8f7582" />

```
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

```
<Figure size 600x400 with 1 Axes><img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/aa97a23b-c105-4088-9788-daae97080fba" />

```
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age Distribution by Survival")
plt.xlabel("Survival (0=No, 1=Yes)")
plt.ylabel("Age")
plt.show()

```
<Figure size 600x400 with 1 Axes><img width="531" height="393" alt="image" src="https://github.com/user-attachments/assets/02879226-a439-4d82-b228-899b143629ef" />

```
plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare by Passenger Class")
plt.show()

```
<Figure size 600x400 with 1 Axes><img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/cea72c04-191e-4ba0-942b-ad58dbccf043" />

```
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

```
<Figure size 800x600 with 2 Axes><img width="706" height="597" alt="image" src="https://github.com/user-attachments/assets/3cb504c5-a257-43cb-a029-526ddfdf3e42" />

```
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```
<Figure size 800x600 with 2 Axes><img width="706" height="597" alt="image" src="https://github.com/user-attachments/assets/36b59611-88ec-4631-becc-ef3bc30ba964" />

```

plt.figure(figsize=(6,3))
sns.boxplot(x=df['Fare'])
plt.title("Outlier Detection in Fare")
plt.show()
```
<Figure size 600x300 with 1 Axes><img width="489" height="316" alt="image" src="https://github.com/user-attachments/assets/40f8ec60-92f8-4ac9-8cce-f17a5a904b30" />

```
print(df['Survived'].value_counts(normalize=True) * 100)
print(df.groupby('Survived')['Age'].mean())
print(df.groupby('Pclass')['Survived'].mean() * 100)
print(df.groupby('Sex')['Survived'].mean() * 100)
df.to_csv("titanic_dataset_EDA.csv", index=False)

```

<img width="910" height="418" alt="image" src="https://github.com/user-attachments/assets/35faae1c-94dd-44ee-9d8e-13b0b7fed891" />




# RESULT
The above code and output for the EDA Analysis using Python

