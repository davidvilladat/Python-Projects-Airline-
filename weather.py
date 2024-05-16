import os
import shutil
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

# Parameters
Countries = {'Latin': ['MDE', 'NVA', 'VVC']}
Years = 5
Save_heatmaps = True
Save_summ = True

# Load and prepare airports data
airports = pd.read_csv("airports.csv")
airports.drop(["id", "ident", "iso_region", "continent", "type", "scheduled_service", "local_code", "home_link", "wikipedia_link", "keywords"], axis=1, inplace=True)

# Filter airports based on the specified countries
request = [airport for country in Countries.values() for airport in country]
airports = airports[airports['iata_code'].isin(request)]
airports.set_index("iata_code", inplace=True)

# Visualize airports on a map using folium
latitude, longitude = airports['latitude_deg'].median(), airports['longitude_deg'].median()
vivaair_map = folium.Map(location=[latitude, longitude], zoom_start=5)

for lat, lng, label in zip(airports.latitude_deg, airports.longitude_deg, airports.index + ' ' + airports.municipality):
    folium.Marker([lat, lng], popup=label).add_to(vivaair_map)

vivaair_map.save("vivaair_map.html")

# Function to get data from Iowa State University - Mesonet
def getdata(stationslist, year=5, df=airports, adjUTC=-5):
    year = max(1, year)  # Ensure year is at least 1

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    url += "".join([f"station={df.at[i, 'gps_code']}&" for i in stationslist])
    url += "data=tmpc&data=drct&data=sknt&data=vsby&data=skyl1&data=wxcodes&"
    
    end_date = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
    url += f"&year1={end_date.year-year}&month1=1&day1=1&year2={end_date.year}&month2={end_date.month}&day2={end_date.day}"
    url += "&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=null&trace=null&direct=yes&report_type=1&report_type=2"
    
    print(url)
    data = pd.read_csv(url)

    # Preprocess data
    data["valid"] = pd.to_datetime(data["valid"])
    data["Year"] = data["valid"].dt.year
    data["Month"] = data["valid"].dt.month
    data["Day"] = data["valid"].dt.day
    data["Hour UTC"] = data["valid"].dt.hour
    data["Hour LT"] = (data["valid"] + datetime.timedelta(hours=adjUTC)).dt.hour
    data["Rain"] = data['wxcodes'].str.contains('RA').fillna(False).astype(int) * 100
    data["Fog-Brume"] = data['wxcodes'].str.contains('FG|BR|HZ').fillna(False).astype(int) * 100
    data["Thunder"] = data['wxcodes'].str.contains('TS').fillna(False).astype(int) * 100
    data["Snow"] = data['wxcodes'].str.contains('SN').fillna(False).astype(int) * 100
    data["Op constraint"] = ((data["Fog-Brume"] == 100) | (data["Thunder"] == 100)).astype(int) * 100

    data.rename(columns={"station": "OACI", "tmpc": "Temperature", "drct": "Wind Dir", "vsby": "H Vsby", "skyl1": "V Vsby", "sknt": "Wind"}, inplace=True)
    data.drop(['valid', 'wxcodes'], axis=1, inplace=True)
    
    df2 = df[['gps_code']].reset_index()
    data = pd.merge(data, df2, how='left', left_on='OACI', right_on='gps_code')
    data.rename(columns={"iata_code": "IATA"}, inplace=True)
    data['IATA'].fillna(data['OACI'], inplace=True)
    data.drop(['gps_code'], axis=1, inplace=True)
    
    return data[['IATA', 'OACI', 'Temperature', 'Wind Dir', 'Wind', 'H Vsby', 'V Vsby', 'Year', 'Month', 'Day', 'Hour UTC', 'Hour LT', 'Rain', 'Fog-Brume', 'Thunder', 'Snow', 'Op constraint']]

data = getdata(request, year=Years)
data

# Create directories for saving heatmaps
if Save_heatmaps or Save_summ:
    dir_path = os.path.join(os.getcwd(), str(datetime.date.today().strftime("%y%m%d")) + " Airport Heatmaps")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# Initialize data structures for storing heatmap data
dataframes, group = {}, {}
columns = ['Temperature', 'Wind', 'Rain', 'Fog-Brume', 'Thunder', 'Snow', 'Op constraint']
columns_title = ['Temperature (°C)', 'Wind (knots)', '% Rain', '% Fog-Brume', '% Thunder', '% Snow', '% Op constraint']
hours = ['Hour UTC', 'Hour LT']

for airport in request:
    for hour in hours:
        for column in columns:
            df = data.loc[data['IATA'] == airport, ['IATA', 'Month', hour, column]]
            grouped = df.groupby(['Month', hour]).mean().unstack().fillna(0).T
            dataframes[(airport, hour, column)] = df
            group[(airport, hour, column)] = grouped.round(2)

# Generate and save heatmaps
vmin, vmax = 0, 12
for airport in request:
    if Save_heatmaps:
        airport_dir = os.path.join(dir_path, airport)
        if os.path.exists(airport_dir):
            shutil.rmtree(airport_dir)
        os.makedirs(airport_dir)

    for hour in hours:
        for column in columns:
            plt.figure(figsize=(10, 8))
            cmap = "afmhot_r" if column in ["Temperature", "Wind"] else "RdYlGn_r"
            ax = sns.heatmap(group[(airport, hour, column)].droplevel(0), cmap=cmap, vmin=vmin, vmax=vmax, annot=True, linewidths=.01)
            ax.tick_params(axis="y", rotation=0)
            ax.set_ylabel(hour, fontsize=10, family="serif")
            ax.set_xlabel("Month", fontsize=10, family="serif")
            ax.set_title(f"{columns_title[columns.index(column)]} - {airport} Airport ({hour.replace('Hour ', '')})", fontsize=15, family="serif")
            
            if Save_heatmaps:
                file_name = f"{hour.replace('Hour ', '')} {columns_title[columns.index(column)]} - {airport} Airport.png"
                plt.savefig(os.path.join(airport_dir, file_name))

# Save summary statistics to an Excel file
if Save_summ:
    summary = data.groupby(['IATA', 'Month', 'Hour LT']).mean().reset_index()
    summary.to_excel("Wx Stats.xlsx", sheet_name="WX", index=False)

# Display the summary DataFrame
summary.groupby(['IATA', 'Month', 'Hour LT']).mean().reset_index()
