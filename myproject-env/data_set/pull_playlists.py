import pandas as pd

# Load your dataset
df = pd.read_csv('high_popularity_spotify_data.csv')

# Define mood classification
def classify_mood(row):
    valence = row['valence']
    energy = row['energy']
    danceability = row['danceability']
    loudness = (row['loudness'])/60 + 1
    speechiness = row['speechiness']
    tempo = (row['tempo'])/200
    # artist = row['track_artist']
    title = row['track_name']

    # this normalizes the ranges to be more like 0-1

    acousticness = row['acousticness'] # probably least helpful

   
    # elif (artist.strip().casefold() == 'kanye west'): 
    #     mood = "Kanye"
    if (acousticness > 0.85 and danceability < 0.5) :
        mood = "Instrumental"
    elif ((valence > 0.6 and valence < 0.75 and energy > 0.6 and tempo < 0.6)):
        mood = 'Happy'
    elif (valence > 0.6 and speechiness < 0.07 and danceability < 0.7):
        mood = 'Strumming' # guitar could be considered hype subclass of it
    elif ((valence > 0.1 and valence < 0.6) and (danceability > 0.4 and danceability < 0.8) and (loudness > 0.8) and (speechiness > 0.03 and speechiness < 0.06) and (tempo > 0.5 and tempo < 0.8)):
        mood = 'Hopeful'
    elif (loudness > 0.8 and danceability > 0.3 and valence > 0.2) or energy > 0.9:
        if (tempo > 0.6 and valence > 0.3) :
            mood = 'Hype'
        else :
            mood = 'Slow Hype'
    elif (valence < 0.4 and danceability < 0.5 and tempo > 0.7):
        mood = "Atmospheric Chill"
    elif ((valence < 0.4 and danceability < 0.5 and speechiness < 0.05) or (energy < 0.6 and danceability < 0.6 and tempo < 0.7)):
        if acousticness > 0.5:
            mood = 'Extra Chill'
        else:
            mood = 'Chill'
    elif (valence > 0.65 and danceability > 0.65 and loudness > 0.85 and tempo < 0.65):
        mood = 'Upbeat Chill'   # fast, chill, yet groovy
    elif ((valence < 0.5 and (energy < 0.5 or danceability < 0.5)and tempo < 0.6)):
        if speechiness > 0.06:
            mood = 'Slow Rap'
        else:
            mood = "Slow"
    elif (acousticness > 0.85 or (danceability > 0.5 and valence > 0.5)):
        mood = "From the Heart"
    elif (danceability > 0.85) :
        mood = "Club Beats"
    
    # what does low valence have in common: they all have high tempo
    elif (valence < 0.2) :
        if (energy < 0.5) :
            mood = 'Slow Hype' # or maybe upbeat chill
        else :
            mood = 'Hype'
    elif (title == 'Jukebox Joints (feat. Joe Fox & Kanye West)') :
        mood = 'Slow Hype'
    elif (title == 'The Devil Wears a Suit and Tie') :
        mood = 'Country'
    elif (title == 'i wanna be your girlfriend') :
        mood = 'Extra Chill'
    else :
        mood = 'Rock'



    # elif () :
    #     mood = 'Unknown'
    
    # elif valence < 0.4 and energy > 0.7 and loudness > -6 and speechiness > 0.4:
    #     mood = 'Angry'
    # elif 0.5 <= valence <= 0.7 and energy < 0.5 and speechiness < 0.4:
    #     mood = 'Romantic'
    # elif valence > 0.6 and energy > 0.8:
    #     mood = 'Excited'
    # elif valence < 0.4 and energy < 0.3 and speechiness < 0.3:
    #     mood = 'Lonely'

    return mood 


# Apply mood classification
df['mood'] = df.apply(classify_mood, axis=1)
df['loudness'] = df['loudness']/60 + 1
df['tempo'] = df['tempo']/200

# Save to new file
df.to_csv('updated_mood_data.csv', index=False)

specific_songs = df[df['mood'] == 'Extra Chill']
# specific_songs = df[df['mood'] == 'Low Happiness']

# Then take a random sample of 50
random_specific_50 = specific_songs
# figure out how preprocess the data
print(random_specific_50[['track_name', 'track_artist', 'energy', 'loudness', 'tempo', 'mood']]) # speechiness , danceability

# 'track_name', 'track_artist', 'valence', 'energy', 'mood'
# Preview
# print(df[['track_name', 'track_artist', 'valence', 'danceability', 'loudness', 'speechiness', 'tempo', 'mood']].sample(50))
# print(df[df['track_name'] == "Disease"])