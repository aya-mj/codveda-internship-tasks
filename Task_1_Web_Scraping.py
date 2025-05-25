import requests
import pandas as pd
from IPython.display import display, Image

# NASA API Key and URLs
api_key = 'I5lR0iYolVkVRE5gNAwbIkTv8qgusvLfBxbIO1r0'
apod_url = f'https://api.nasa.gov/planetary/apod?api_key={api_key}'

# Fetching APOD Data
response = requests.get(apod_url)
if response.status_code == 200:
    My_data = response.json()
    if 'url' in My_data:
        print(f"Title : {My_data['title']}")
        print(f"Explanation : {My_data['explanation']}")
        print(f"Image url : {My_data['url']}")
        display(Image(My_data['url']))
    else:
        print('No image found')
else:
    print(f"Failed to retrieve APOD data: {response.status_code}")

# Fetching NEO Data
import requests
import pandas as pd

api_key = 'I5lR0iYolVkVRE5gNAwbIkTv8qgusvLfBxbIO1r0'
neo_url = f'https://api.nasa.gov/neo/rest/v1/feed?api_key={api_key}'

neo_response = requests.get(neo_url)
if neo_response.status_code == 200:
    neo_data = neo_response.json()
    asteroid = []
    for date, asteroid_list in neo_data['near_earth_objects'].items():
        for asteroid_data in asteroid_list:
            asteroid.append({
                'Asteroid ID': asteroid_data.get('id', ''),
                'Asteroid Name': asteroid_data.get('name', ''),
                'The Minimal estimated diameter (km)': asteroid_data['estimated_diameter']['kilometers'].get('estimated_diameter_min', None),
                'Absolute magnitude': asteroid_data.get('absolute_magnitude_h', None),
                'Relative_velocity (km/s)': asteroid_data['close_approach_data'][0].get('relative_velocity', {}).get('kilometers_per_second', None) 
                if asteroid_data['close_approach_data'] else None
            })

    # Storing data into DataFrame
    df = pd.DataFrame(asteroid)
    print(df)
    
    # Saving to CSV
    df.to_csv('asteroids_data.csv', index=False)
else:
    print(f"Failed to retrieve NEO data: {neo_response.status_code}")
