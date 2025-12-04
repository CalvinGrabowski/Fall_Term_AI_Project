import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from spotipy import Spotify
import joblib
import requests
import time
import pandas as pd

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".." , "..", "secret.env"))
client_id = os.getenv("SPOTIPY_CLIENT_ID") 
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET") 
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")




auth_manager = SpotifyOAuth( client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope="playlist-read-private playlist-read-collaborative" )
#  print (redirect_uri)
sp = Spotify(auth_manager=auth_manager)

model = joblib.load("mood_model.pkl") 
le = joblib.load("label_encoder.pkl")

features = ['valence', 'energy', 'danceability', 'acousticness', 'loudness', 'speechiness']

# this finds the playlist and allows us to group the songs by themselves
# playlist_link = "https://open.spotify.com/playlist/7vWVX36xMHGwi1f9L9OX61" # this is roxane playlist
playlist_link = "https://open.spotify.com/playlist/22vzWz8jg0xrIqBoE59ULO" # this is my throwback's playlist
playlist_id = playlist_link.split("/")[-1].split("?")[0]
results = sp.playlist_tracks(playlist_id) 
tracks = results['items']






# After you've built the client and fetched tracks:
for item in tracks : # [:1]:
    track = item['track']
    track_id = track['id']
    track_name = track['name']
    artist = track['artists'][0]['name']
    print(f"Song: {track_name} | Artist: {artist} | Track['id']: {track_id}")

    try:
        
        # audio_features = sp.audio_features(track_id)[0] no longer a feature on spotipy
        


        url = "https://track-analysis.p.rapidapi.com/pktx/analysis"
        params = {
            "song": track_name,
            "artist": artist # optional field but recommended for accuracy
        }

        headers = {
            "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
            "X-RapidAPI-Host": "track-analysis.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=params)

        print("Status code:", response.status_code)
        print("Response JSON:", response.json())


        data = response.json()
        # print("Response JSON:", data)
        
        
# features = ['valence', 'energy', 'danceability', 'acousticness', 'loudness', 'speechiness']
        # make all the data get prepped into 0-1 range, and order it in the same order for the ai to analysis
        def preprocess_features(data):
            raw_loudness = data.get("loudness", "-30 dB")
            loudness_val = float(raw_loudness.replace(" dB", ""))

            return {
                "valence": data.get("happiness", 50)/100.0,
                "energy": data.get("energy", 50)/100.0,
                "danceability": data.get("danceability", 50)/100.0,
                "acousticness": data.get("acousticness", 50)/100.0,
                "loudness": (loudness_val / 60.0) + 1,  # normalize dB
                "speechiness": data.get("speechiness", 50)/100.0
                # "tempo": data.get("tempo", 0) / 200.0,               # not needed, not tested with
            }

        X_song = pd.DataFrame([preprocess_features(data)])
        pred = model.predict(X_song)
        mood = le.inverse_transform(pred)[0]
        print("Feature vector:\n", X_song.round(4).to_string(index=False))
        print(f"Predicted mood: {mood}")

        time.sleep(1)


        # if audio_features:
        #     # Grab just the tempo
        #     print("step 3")
        #     tempo = audio_features.get("tempo")
        #     print(f"Tempo for {track_name}: {tempo}")

        #     # If you still want to build the full feature vector:
        #     X_song = np.array([[audio_features.get(f, 0) for f in features]])
        #     pred = model.predict(X_song)
        #     mood = le.inverse_transform(pred)[0]
        #     print(f"Predicted mood: {mood}")
        # else:
        #     print(f"⚠️ No audio features for {track_name}")
    except Exception as e:
        print(f"⚠️ Failed to fetch features for {track_name}: {e}")