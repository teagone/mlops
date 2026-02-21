import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - CANCER PATIENT DATASET")
print("=" * 60)

df = pd.read_csv('cancer patient data sets.csv')

# Basic Information
print("\n1. DATASET OVERVIEW")
print("-" * 60)
print(f"Shape: {df.shape} (rows, columns)")
print(f"\nColumn Names:")
print(df.columns.tolist())
print(f"\nFirst few rows:")
print(df.head())

# Data Types and Missing Values
print("\n2. DATA TYPES AND MISSING VALUES")
print("-" * 60)
print(df.dtypes)
print(f"\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

# Summary Statistics
print("\n3. SUMMARY STATISTICS")
print("-" * 60)
print(df.describe())

# Target Variable Analysis
print("\n4. TARGET VARIABLE ANALYSIS (Level)")
print("-" * 60)
if 'Level' in df.columns:
    print(df['Level'].value_counts())
    print(f"\nPercentage distribution:")
    print(df['Level'].value_counts(normalize=True) * 100)

# Gender Analysis
print("\n5. GENDER DISTRIBUTION")
print("-" * 60)
if 'Gender' in df.columns:
    gender_counts = df['Gender'].value_counts()
    print(gender_counts)
    print(f"\nNote: Gender appears to be encoded (1 or 2)")

# Age Analysis
print("\n6. AGE STATISTICS")
print("-" * 60)
if 'Age' in df.columns:
    print(f"Age Statistics:")
    print(df['Age'].describe())
    print(f"\nAge Range: {df['Age'].min()} - {df['Age'].max()}")

# Risk Factors Analysis
print("\n7. RISK FACTORS OVERVIEW")
print("-" * 60)
risk_factors = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 
                'Genetic Risk', 'chronic Lung Disease', 'Obesity', 'Smoking', 
                'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 
                'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
                'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']

# Check which risk factors exist in the dataset
existing_risk_factors = [col for col in risk_factors if col in df.columns]
print(f"Found {len(existing_risk_factors)} risk factor columns")
print(f"\nRisk Factor Statistics:")
print(df[existing_risk_factors].describe())

# Create Visualizations
print("\n8. GENERATING VISUALIZATIONS...")
print("-" * 60)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Target Variable Distribution
ax1 = plt.subplot(3, 3, 1)
if 'Level' in df.columns:
    df['Level'].value_counts().plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title('Distribution of Cancer Risk Levels', fontsize=12, fontweight='bold')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

# 2. Age Distribution
ax2 = plt.subplot(3, 3, 2)
if 'Age' in df.columns:
    df['Age'].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('Age Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

# 3. Gender Distribution
ax3 = plt.subplot(3, 3, 3)
if 'Gender' in df.columns:
    df['Gender'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.title('Gender Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Gender (1 or 2)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

# 4. Age by Risk Level
ax4 = plt.subplot(3, 3, 4)
if 'Age' in df.columns and 'Level' in df.columns:
    df.boxplot(column='Age', by='Level', ax=ax4)
    plt.title('Age Distribution by Risk Level', fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Risk Level')
    plt.ylabel('Age')

# 5. Correlation Heatmap (select numeric columns)
ax5 = plt.subplot(3, 3, 5)
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    # Select a subset of columns for better visualization
    cols_to_plot = numeric_cols[:15]  # First 15 numeric columns
    corr_matrix = df[cols_to_plot].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax5, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap (Top 15 Features)', fontsize=12, fontweight='bold')

# 6. Risk Factors Mean by Level
ax6 = plt.subplot(3, 3, 6)
if 'Level' in df.columns and len(existing_risk_factors) > 0:
    level_means = df.groupby('Level')[existing_risk_factors[:10]].mean()
    level_means.T.plot(kind='bar', ax=ax6, width=0.8)
    plt.title('Average Risk Factor Scores by Level', fontsize=12, fontweight='bold')
    plt.xlabel('Risk Factors')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Top Risk Factors Distribution
ax7 = plt.subplot(3, 3, 7)
if len(existing_risk_factors) > 0:
    top_risk = existing_risk_factors[0]
    df[top_risk].hist(bins=20, color='coral', edgecolor='black', ax=ax7)
    plt.title(f'Distribution of {top_risk}', fontsize=12, fontweight='bold')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

# 8. Smoking vs Risk Level
ax8 = plt.subplot(3, 3, 8)
if 'Smoking' in df.columns and 'Level' in df.columns:
    smoking_by_level = pd.crosstab(df['Level'], df['Smoking'])
    smoking_by_level.plot(kind='bar', stacked=True, ax=ax8, colormap='viridis')
    plt.title('Smoking Levels by Risk Category', fontsize=12, fontweight='bold')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Smoking Score', bbox_to_anchor=(1.05, 1), loc='upper left')

# 9. Summary Statistics Boxplot
ax9 = plt.subplot(3, 3, 9)
if len(existing_risk_factors) > 0:
    # Select a few key risk factors for boxplot
    key_factors = existing_risk_factors[:5]
    df_melted = df.melt(id_vars=['Level'] if 'Level' in df.columns else [], 
                       value_vars=key_factors, 
                       var_name='Risk Factor', 
                       value_name='Score')
    if 'Level' in df.columns:
        sns.boxplot(data=df_melted, x='Risk Factor', y='Score', hue='Level', ax=ax9)
    else:
        sns.boxplot(data=df_melted, x='Risk Factor', y='Score', ax=ax9)
    plt.title('Risk Factor Scores Distribution', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    if 'Level' in df.columns:
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('cancer_patient_eda.png', dpi=300, bbox_inches='tight')
print("Visualizations saved to 'cancer_patient_eda.png'")

# Additional Analysis
print("\n9. ADDITIONAL INSIGHTS")
print("-" * 60)

# Check for any patterns
if 'Level' in df.columns:
    print("\nAverage Age by Risk Level:")
    print(df.groupby('Level')['Age'].mean())
    
    print("\nAverage Risk Factor Scores by Level:")
    if len(existing_risk_factors) > 0:
        print(df.groupby('Level')[existing_risk_factors].mean().round(2))

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
