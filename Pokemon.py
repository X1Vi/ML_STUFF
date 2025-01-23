import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Load your dataset (replace 'your_file.csv' with your file name)
df = pd.read_csv("pokemon.csv")  # If it's a CSV file

# Select the required columns
columns_to_keep = ['name', 'type1', 'type2', 'attack', 'defense', 'hp', 'speed', 'generation', 'is_legendary']
filtered_df = df[columns_to_keep]
print(filtered_df)

# Scatter plot of HP vs Speed with color-coding by Generation
plt.figure(figsize=(10, 6))  # Adjust the size of the plot to give more space
sns.scatterplot(x='speed', y='hp', data=filtered_df, hue='generation', palette='viridis', alpha=0.7)

# Add titles and labels
plt.title('Pokemon HP vs Speed')
plt.xlabel('Speed')
plt.ylabel('HP')

# Adjust the legend to appear outside the plot area
plt.legend(title='Generation', bbox_to_anchor=(1.05, 1), loc='upper left')

# Automatically adjust plot layout to fit everything
plt.tight_layout()

# Display the plot
plt.show()








# Scatter plot of HP vs Speed with color-coding by Generation
plt.figure(figsize=(10, 6))  # Adjust the size of the plot to give more space
sns.scatterplot(x='attack', y='hp', data=filtered_df, hue='generation', palette='viridis', alpha=0.7)

# Add titles and labels
plt.title('Pokemon Attack vs HP')
plt.xlabel('Attack')
plt.ylabel('HP')

# Adjust the legend to appear outside the plot area
plt.legend(title='Generation', bbox_to_anchor=(1.05, 1), loc='upper left')

# Automatically adjust plot layout to fit everything
plt.tight_layout()

# Display the plot
plt.show()





# K means clustering

columns_to_keep = ['name', 'type1', 'type2', 'attack', 'defense', 'hp', 'speed', 'generation', 'is_legendary']
filtered_df = df[columns_to_keep]

# 1. Select features for clustering
features = ['attack', 'defense', 'hp', 'speed']
X = filtered_df[features]

# 2. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
filtered_df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. Scatter plot of HP vs Speed with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='speed', y='hp', data=filtered_df, hue='cluster', palette='Set2', alpha=0.7)

# Add titles and labels
plt.title('Pokemon HP vs Speed with Clusters')
plt.xlabel('Speed')
plt.ylabel('HP')

# Adjust the legend to appear outside the plot area
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Automatically adjust plot layout to fit everything
plt.tight_layout()

# Display the plot
plt.show()


#3D Plot
# Assuming 'filtered_df' is your DataFrame containing the relevant columns
# Assuming 'filtered_df' is your DataFrame containing the relevant columns
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define a color map for generations
generation_colors = {
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'purple',
    5: 'orange',
    6: 'cyan',
    7: 'magenta'
}

# Map generations to colors
filtered_df['color'] = filtered_df['generation'].map(generation_colors)

# Plot each generation with its corresponding color
for generation, color in generation_colors.items():
    gen_data = filtered_df[filtered_df['generation'] == generation]
    ax.scatter(gen_data['speed'], gen_data['hp'], gen_data['attack'], c=color, label=f'Generation {generation}')

# Add titles and labels
ax.set_title('3D Scatter Plot of Pokémon Attack, HP, and Speed')
ax.set_xlabel('Speed')
ax.set_ylabel('HP')
ax.set_zlabel('Attack')

# Add legend with adjusted position
ax.legend(title='Generation', loc='upper left', bbox_to_anchor=(-0.1, 1))

# Display the plot
plt.show()



