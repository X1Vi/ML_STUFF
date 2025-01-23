from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression



# prediction model
df = pd.read_csv('pokemon.csv')

# Handle missing values
df['type2'] = df['type2'].fillna('None')
df['attack'] = df['attack'].fillna(df['attack'].mean())
df['defense'] = df['defense'].fillna(df['defense'].mean())
df['hp'] = df['hp'].fillna(df['hp'].mean())
df['speed'] = df['speed'].fillna(df['speed'].mean())

# Feature engineering: Extract name length
df['name_length'] = df['name'].apply(len)

# Define features and target
X = df[['name', 'type1', 'type2', 'attack', 'defense', 'hp', 'speed', 'name_length']]
y = df['is_legendary']

# Preprocessing for categorical features
categorical_features = ['name', 'type1', 'type2']
numeric_features = ['attack', 'defense', 'hp', 'speed', 'name_length']

# Create transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Add handle_unknown='ignore'
    ])

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Example input data (dummy values)
test_data = pd.DataFrame({
    'name': ['Pikachu', 'Charizard', 'Bulbasaur'],
    'type1': ['Electric', 'Fire', 'Grass'],
    'type2': ['None', 'Flying', 'Poison'],
    'attack': [55, 84, 49],
    'defense': [40, 78, 49],
    'hp': [35, 78, 45],
    'speed': [90, 100, 45],
    'name_length': [8, 8, 9]  # Length of the name
})

# Adding legendary Pokémon (Mewtwo)
legendary_data = pd.DataFrame({
    'name': ['Mewtwo', 'Lugia', 'Rayquaza', 'Articuno', 'Moltres'],
    'type1': ['Psychic', 'Psychic', 'Dragon', 'Ice', 'Fire'],
    'type2': ['None', 'Flying', 'Flying', 'Flying', 'Flying'],
    'attack': [154, 90, 150, 100, 100],
    'defense': [60, 130, 90, 85, 90],
    'hp': [106, 106, 105, 90, 90],
    'speed': [130, 110, 95, 85, 90],
    'name_length': [7, 4, 8, 8, 7]  # Length of the name
})

# Append legendary Pokémon to the test_data
test_data = pd.concat([test_data, legendary_data], ignore_index=True)

# Display the updated DataFrame
print(test_data)

# Predict on the new test data
predictions = model.predict(test_data)

# Display the predicted results
for name, prediction in zip(test_data['name'], predictions):
    print(f"Prediction for {name}: {'Legendary' if prediction == 1 else 'Not Legendary'}")
