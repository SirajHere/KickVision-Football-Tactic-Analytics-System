import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data=pd.read_csv('/Users/shubh/Downloads/training.csv')
sns.displot(data=data, x='itching', kind='hist', bins=20)
sns.displot(data=data, x='skin_rash', kind='hist', bins=20)
sns.displot(data=data, x='stomach_pain', kind='hist', bins=20)
sns.displot(data=data, x='cough', kind='hist', bins=20)
sns.displot(data=data, x='headache', kind='hist', bins=20)
sns.displot(data=data, x='nausea', kind='hist', bins=20)
sns.displot(data=data, x='vomiting', kind='hist', bins=20)
sns.displot(data=data, x='abdominal_pain', kind='hist', bins=20)
sns.displot(data=data, x='acidity', kind='hist', bins=20)
sns.displot(data=data, x='congestion', kind='hist', bins=20)
selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'stomach_pain'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=subset_df, x='prognosis', hue=specific_symptom, order=selected_diseases)
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Disease')
plt.ylabel(f'Occurrence of {specific_symptom}')
plt.legend(title='Symptom', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'abdominal_pain'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=subset_df, x='prognosis', hue=specific_symptom, order=selected_diseases)
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Disease')
plt.ylabel(f'Occurrence of {specific_symptom}')
plt.legend(title='Symptom', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'vomiting'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=subset_df, x='prognosis', hue=specific_symptom, order=selected_diseases)
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Disease')
plt.ylabel(f'Occurrence of {specific_symptom}')
plt.legend(title='Symptom', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'cramps'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=subset_df, x='prognosis', hue=specific_symptom, order=selected_diseases)
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Disease')
plt.ylabel(f'Occurrence of {specific_symptom}')
plt.legend(title='Symptom', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'skin_peeling'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=subset_df, x='prognosis', hue=specific_symptom, order=selected_diseases)
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Disease')
plt.ylabel(f'Occurrence of {specific_symptom}')
plt.legend(title='Symptom', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'yellow_crust_ooze'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(data=subset_df, x='prognosis', hue=specific_symptom, order=selected_diseases)
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Disease')
plt.ylabel(f'Occurrence of {specific_symptom}')
plt.legend(title='Symptom', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'vomiting'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()


selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'acidity' 
# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'abdominal_pain'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'stomach_pain'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'headache'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'cramps'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

selected_diseases = ['Gastroenteritis', 'GERD', 'Peptic ulcer diseae', 'Alcoholic hepatitis']
specific_symptom = 'nausea'

# Filter the dataset to include only the selected diseases
subset_df = data[data['prognosis'].isin(selected_diseases)]

# Pivot the data to get a matrix of occurrences of the specific symptom for each disease
heatmap_data = subset_df.pivot_table(index='prognosis', columns=specific_symptom, aggfunc='size', fill_value=0)

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
plt.title(f'Occurrence of {specific_symptom} for Selected Diseases')
plt.xlabel('Symptom Occurrence (0: Absent, 1: Present)')
plt.ylabel('Disease')
plt.xticks(ticks=[0.5, 1.5], labels=['Absent', 'Present'])  # Customize x-axis ticks
plt.show()
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random

random_columns = random.sample(list(data.columns), 10)
selected_columns = random_columns + ['prognosis']

# Subset the data with the selected columns
data_subset = data[selected_columns].copy()

# Encode the 'prognosis' column

# Create pairplot
sns.pairplot(data_subset, hue='prognosis', diag_kind='kde')

# Plot the pairplot

plt.show()
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
newdata= LabelEncoder()
data['prognosis'] = newdata.fit_transform(data['prognosis'])
random_columns = random.sample(list(data.columns), 10)
selected_columns = random_columns + ['prognosis']

# Subset the data with the selected columns
data_subset = data[selected_columns].copy()
correlation_matrix = data_subset.corr()
fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu',
                title='Correlation Heatmap of Random 10 Symptoms (Data set )')
fig.show()
