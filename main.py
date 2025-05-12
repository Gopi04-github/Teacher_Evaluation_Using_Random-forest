import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Generate mock dataset
np.random.seed(42)
sections = ['A', 'B', 'C', 'D', 'E']
students_per_section = 10
data = []
roll_number = 1

for section in sections:
    for _ in range(students_per_section):
        theory = np.random.randint(0, 11)
        logical = np.random.randint(0, 11)
        analytical = np.random.randint(0, 11)
        marks = {'Theory': theory, 'Logical': logical, 'Analytical': analytical}
        best_subject = max(marks, key=marks.get)
        data.append([roll_number, section, theory, logical, analytical, best_subject])
        roll_number += 1

df = pd.DataFrame(data, columns=['Roll_Number', 'Section', 'Theory_Mark', 'Logical_Mark', 'Analytical_Mark', 'Result'])

# Save dataset
df.to_csv("teacher_ml_dataset.csv", index=False)

# Model Training
X = df[['Theory_Mark', 'Logical_Mark', 'Analytical_Mark']]
y = df['Result']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# New sample prediction
new_sample = pd.DataFrame({
    'Theory_Mark': [8],
    'Logical_Mark': [5],
    'Analytical_Mark': [7]
})
predicted_class = le.inverse_transform(rf.predict(new_sample))
print("\nPredicted Best Subject for New Sample:", predicted_class[0])

# Plot: Distribution of Strongest Subject
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Result', palette='Set2')
plt.xlabel('Strongest Subject')
plt.ylabel('Number of Students')
plt.title('Distribution of Students by Strongest Subject')
plt.tight_layout()
plt.show()

# Section-wise Average Scores
section_avg = df.groupby('Section')[['Theory_Mark', 'Logical_Mark', 'Analytical_Mark']].mean().reset_index()
print("\nSection-wise Average Marks:\n", section_avg)

# Plot: Section-wise Average Marks
section_avg_melted = section_avg.melt(id_vars='Section', var_name='Subject', value_name='Average_Mark')
plt.figure(figsize=(10, 6))
sns.barplot(data=section_avg_melted, x='Section', y='Average_Mark', hue='Subject', palette='Paired')
plt.title("Average Marks by Section and Subject")
plt.ylabel("Average Marks")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Identify top section for each subject
top_sections = section_avg.set_index('Section').idxmax()
print("\nTop Sections for Each Subject:\n", top_sections)
