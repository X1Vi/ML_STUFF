from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing, SMOTE, and model using imblearn pipeline
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42))
])

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
    'name': [
        'Mewtwo', 'Lugia', 'Rayquaza', 'Articuno', 'Moltres',
        'Zapdos', 'Ho-Oh', 'Groudon', 'Kyogre', 'Dialga',
        'Palkia', 'Giratina', 'Reshiram', 'Zekrom', 'Xerneas',
        'Yveltal', 'Zygarde', 'Solgaleo', 'Lunala', 'Eternatus'
    ],
    'type1': [
        'Psychic', 'Psychic', 'Dragon', 'Ice', 'Fire',
        'Electric', 'Fire', 'Ground', 'Water', 'Steel',
        'Water', 'Ghost', 'Dragon', 'Dragon', 'Fairy',
        'Dark', 'Dragon', 'Psychic', 'Ghost', 'Poison'
    ],
    'type2': [
        'None', 'Flying', 'Flying', 'Flying', 'Flying',
        'Flying', 'Flying', 'None', 'None', 'Dragon',
        'Dragon', 'Dragon', 'Fire', 'Electric', 'None',
        'Flying', 'Ground', 'Steel', 'Psychic', 'Dragon'
    ],
    'attack': [
        154, 90, 150, 100, 100,
        125, 130, 150, 150, 120,
        120, 120, 120, 120, 131,
        131, 100, 137, 137, 145
    ],
    'defense': [
        60, 130, 120, 85, 90,
        90, 90, 140, 90, 120,
        100, 120, 100, 100, 98,
        95, 121, 107, 107, 95
    ],
    'hp': [
        106, 106, 155, 90, 90,
        90, 106, 100, 100, 100,
        90, 150, 100, 100, 126,
        126, 108, 137, 137, 140
    ],
    'speed': [
        130, 110, 95, 85, 90,
        100, 90, 90, 90, 90,
        100, 90, 90, 90, 99,
        99, 95, 97, 97, 130
    ],
    'name_length': [
        len('Mewtwo'), len('Lugia'), len('Rayquaza'), len('Articuno'), len('Moltres'),
        len('Zapdos'), len('Ho-Oh'), len('Groudon'), len('Kyogre'), len('Dialga'),
        len('Palkia'), len('Giratina'), len('Reshiram'), len('Zekrom'), len('Xerneas'),
        len('Yveltal'), len('Zygarde'), len('Solgaleo'), len('Lunala'), len('Eternatus')
    ]
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