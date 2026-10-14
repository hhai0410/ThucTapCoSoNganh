import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
df = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")
'''
print("Shape:", df.shape)
df.info()

print("\n=== thong ke mo ta  ===")
print(df.describe())

print("\n=== phan bo nhan ===")
print(df['stroke'].value_counts())
print(df['stroke'].value_counts(normalize=True) * 100)


print("\n=== khong co du lieu ===")
print(df.isnull().sum())

# 6. Phân biệt numerical vs categorical (gợi ý)
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                    'work_type', 'Residence_type', 'smoking_status']

print("\nNumerical:", numerical_cols)
print("Categorical:", categorical_cols)
'''

# Histogram + đường mật độ (KDE)
sns.histplot(data=df, x='age', kde=True)
plt.title('Phân phối Age')

# Boxplot
sns.boxplot(data=df, x='bmi')
plt.show()

# hoặc so sánh theo target
sns.boxplot(data=df, x='stroke', y='age')
# Countplot
sns.countplot(data=df, x='gender', order=df['gender'].value_counts().index)


'''
sns.histplot(df["age"].dropna(), kde=True, color="#3498db", bins=30)
plt.axvline(df["age"].median(), color="red", linestyle="--", label="Median")

sns.boxplot(data=df, x="stroke", y="age", hue="stroke",palette=["#2ecc71", "#e74c3c"], legend=False)

sns.countplot(data=df, x="gender", order=df["gender"].value_counts().index,palette="Set2", edgecolor="black")

sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)

sns.kdeplot(data=df[df["stroke"]==0], x="age", fill=True, alpha=0.4, label="No Stroke", color="#2ecc71")
sns.kdeplot(data=df[df["stroke"]==1], x="age", fill=True, alpha=0.5, label="Stroke", color="#e74c3c")

plt.scatter(df.loc[df.stroke==0, "avg_glucose_level"], df.loc[df.stroke==0, "bmi"],
            c="#2ecc71", alpha=0.35, s=18, label="No Stroke")
plt.scatter(df.loc[df.stroke==1, "avg_glucose_level"], df.loc[df.stroke==1, "bmi"],
            c="#e74c3c", alpha=0.75, s=45, label="Stroke")

ct = pd.crosstab(df["hypertension"], df["stroke"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], edgecolor="black")
'''