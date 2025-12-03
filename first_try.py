import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib 

# Step 1: Load the dataset
df = pd.read_csv('updated_mood_data.csv')

# Step 2: Select features and target
features = ['valence', 'energy', 'danceability', 'acousticness', 'loudness', 'speechiness']
X = df[features]
y = df['mood']

# Step 3: Encode mood labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Step 5: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 7: Predict mood for a new song
new_song = pd.DataFrame([{
    'valence': 0.65,
    'energy': 0.7,
    'danceability': 0.8,
    'acousticness': 0.2,
    'loudness': -5.0,
    'speechiness': 0.05
}])

predicted_label = model.predict(new_song)
print("Predicted mood:", le.inverse_transform(predicted_label)[0])

# Save the trained model
joblib.dump(model, 'mood_model.pkl')

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')

