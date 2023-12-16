import pandas as pd
from datetime import datetime

def convert_date(date_str):
    try:
        # Try to parse the date assuming dd/mm/yyyy format
        return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
    except ValueError:
        # If it fails, assume dd/mm/yy format
        return datetime.strptime(date_str, '%d/%m/%y').strftime('%Y-%m-%d')

# Read the Football DataFrame
football_df = pd.read_csv('./epl-training.csv')
football_df['Date'] = football_df['Date'].apply(convert_date)

# Dictionary mapping team to location
team_location_map = {
    'Hull': 'Hull',
    'Fulham': 'London',
    "Nott'm Forest": 'West_Bridgford',
    'Derby': 'Derby',
    'Leeds': 'Leeds',
    'Arsenal': 'London',
    'Portsmouth': 'Portsmouth',
    'Sheffield United': 'Sheffield',
    'Blackpool': 'Blackpool',
    'QPR': 'London',
    'Bolton': 'Horwich',
    'Cardiff': 'Cardiff',
    'Birmingham': 'Birmingham',
    'Charlton': 'London',
    'Luton': 'Luton',
    'Chelsea': 'London',
    'Newcastle': 'Newcastle_upon_Tyne',
    'Blackburn': 'Blackburn',
    'Bournemouth': 'Bournemouth',
    'Brighton': 'Brighton',
    'Leicester': 'Leicester',
    'Aston Villa': 'Birmingham',
    'Liverpool': 'Liverpool',
    'Crystal Palace': 'London',
    'Swansea': 'Swansea',
    'Coventry': 'Coventry',
    'Wolves': 'Wolverhampton',
    'Man City': 'Manchester',
    'West Ham': 'London',
    'Reading': 'Reading',
    'Southampton': 'Southampton',
    'Burnley': 'Burnley',
    'Man United': 'Stretford',
    'Ipswich': 'Ipswich',
    'Norwich': 'Norwich',
    'Tottenham': 'London',
    'Bradford': 'Bradford',
    'Middlesbrough': 'Middlesbrough',
    'West Brom': 'West_Bromwich',
    'Stoke': 'Stoke-on-Trent',
    'Wigan': 'Wigan',
    'Everton': 'Liverpool',
    'Watford': 'Watford',
    'Brentford': 'Brentford',
    'Sunderland': 'Sunderland',
    'Huddersfield': 'Huddersfield'
}  

# New DataFrame to store combined data
combined_df = pd.DataFrame()

# Iterate through each row of the football DataFrame
for index, row in football_df.iterrows():
    # Map team to location
    location = team_location_map.get(row['HomeTeam'])

    # Construct weather CSV filename
    weather_csv = f'./{location}.csv'

    # Read weather data
    try:
        weather_df = pd.read_csv(weather_csv, skiprows=3)
        weather_df['time'] = pd.to_datetime(weather_df['time'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')

        # Find matching date row
        matching_weather = weather_df[weather_df['time'] == row['Date']]

        if not matching_weather.empty:
            # Combine the data
            combined_row = pd.concat([row, matching_weather.iloc[0]]).to_frame().T
            combined_df = pd.concat([combined_df, combined_row], ignore_index=True)
        else:
            print('no match')

    except FileNotFoundError:
        print(f"Weather file not found for location: {location}")


# Output the combined DataFrame to a new CSV file
combined_df.to_csv('./epl-training-with-weather.csv', mode='a', header=False, index=False)

