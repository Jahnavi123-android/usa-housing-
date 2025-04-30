import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

file_path = 'C:\\Users\\AK\\OneDrive\\Desktop\\ml\\Project 1\\USA_Housing (1).csv'
# Read the CSV file
data = pd.read_csv(file_path)
# Print the first few rows
print(data.head())
print(data.info())
sns.pairplot(data)
plt.show()

plt.plot(data['Avg. Area Income'])
plt.show()

sns.distplot(data['Avg. Area Number of Bedrooms'])
plt.show()

data.drop('Address', axis=1, inplace=True)
print(data.head())

correlation = data.corr()
# Plot the heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - USA Housing')
plt.show()

print(data.corr())
sns.heatmap(data.corr())
plt.show()
sns.distplot(data['Price'])
data.columns
x = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']]
y = data[['Price']]
plt.plot(x,y)
plt.show()

# Convert 'Price' into a classification target
median_price = data['Price'].median()
data['Expensive'] = np.where(data['Price'] > median_price, 1, 0)

# Features and new target
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Expensive']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot metrics
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

plt.figure(figsize=(8,6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.title('Model Performance Metrics')
plt.ylim(0,1)
plt.ylabel('Score')
plt.show()

sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')


try:
    
    prediction = data.predict()  # this will cause AttributeError
except AttributeError as e:
    print("AttributeError occurred:", e)
    # You can add specific handling here, like fallback code
    # For example:
    if 'predict' in str(e):
        print("It seems you tried to call 'predict' on a DataFrame. Check your object!")