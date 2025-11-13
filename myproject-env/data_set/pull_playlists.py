import pandas as pd

# Load your dataset
df = pd.read_csv('high_popularity_spotify_data.csv')

# Define mood classification
def classify_mood(row):
    valence = row['valence']
    energy = row['energy']
    danceability = row['danceability']
    acousticness = row['acousticness']
    loudness = row['loudness']
    speechiness = row['speechiness']

    if valence > 0.7 and energy > 0.6:
        return 'Happy'
    elif valence < 0.3 and energy < 0.5:
        return 'Sad'
    elif acousticness > 0.5 and energy < 0.6 and valence > 0.4:
        return 'Chill'
    elif valence < 0.3 and energy > 0.7 and loudness < -5:
        return 'Angry'
    elif 0.4 < valence < 0.7 and energy < 0.5 and acousticness > 0.4:
        return 'Romantic'
    elif energy > 0.8 and danceability > 0.7 and loudness > -6:
        return 'Hype'
    else:
        return 'Neutral'

# Apply mood classification
df['mood'] = df.apply(classify_mood, axis=1)

# Save to new file
df.to_csv('updated_mood_data.csv', index=False)

# Preview
print(df[['track_name', 'track_artist', 'valence', 'energy', 'mood']].head())