# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:06:43 2024

@author: crist
"""

airports = {
    'NBJ': {'coords': (-8.85837, 13.2312), 'name': 'Luanda (Hub)'},
    'LIS': {'coords': (38.7704, -9.1291), 'name': 'Lisbon'},
    'GRU': {'coords': (-23.6273, -46.6566), 'name': 'Sao Paulo'},
    'JNB': {'coords': (-26.1392, 28.246), 'name': 'Johannesburg'},
    'LOS': {'coords': (6.5779, 3.3212), 'name': 'Lagos'},
    'BZV': {'coords': (-4.2517, 15.253), 'name': 'Brazzaville'},
    'MPM': {'coords': (-25.9208, 32.5721), 'name': 'Maputo'},
    'ACC': {'coords': (5.6052, -0.1668), 'name': 'Accra'},
    'PNR': {'coords': (-4.8160, 11.8869), 'name': 'Pointe-Noire'},
    'CPT': {'coords': (-33.9648, 18.6017), 'name': 'Cape Town'},
    'FIH': {'coords': (-4.3858, 15.4446), 'name': 'Kinshasa'}, 
    'TMS': {'coords': (0.3782, 6.7126), 'name': 'São Tomé'},
    'WDH': {'coords': (-22.4799, 17.4709), 'name': 'Windhoek'},
    'NBO': {'coords': (-1.3192, 36.9275), 'name': 'Nairobi (Potential)'},
    'BOM': {'coords': (19.0896, 72.8656), 'name': 'Mumbai (Potential)'},
    'EZE': {'coords': (-34.8222, -58.5358), 'name': 'Buenos Aires (Potential)'},
    'LHR': {'coords': (51.4700, -0.4543), 'name': 'London (Potential)'},
    'MIA': {'coords': (25.7959, -80.2870), 'name': 'Miami (Potential)'},
    'CDG': {'coords': (49.0097, 2.5479), 'name': 'Paris (Potential)'},
    'FRA': {'coords': (50.0379, 8.5622), 'name': 'Frankfurt (Potential)'},
    'SGN': {'coords': (10.8185, 106.6519), 'name': 'Ho Chi Minh City (Potential)'},
    'BKK': {'coords': (13.689999, 100.750112), 'name': 'Bangkok (Potential)'},
    'CAN': {'coords': (23.3924, 113.2988), 'name': 'Guangzhou (Potential)'},
    'SCL': {'coords': (-33.393, -70.786), 'name': 'Santiago (Potential)'},
    'DXB': {'coords': (25.2528, 55.3644), 'name': 'Dubai (Potential)'},
    'DOH': {'coords': (25.2611, 51.5651), 'name': 'Doha (Potential)'},
    'AUH': {'coords': (24.4330, 54.6511), 'name': 'Abu Dhabi (Potential)'},
    'CAI': {'coords': (30.1219, 31.4056), 'name': 'Cairo (Potential)'},
    'DSS': {'coords': (14.7405, -17.4902), 'name': 'Dakar (Potential)'},
    'DAR': {'coords': (-6.8781, 39.2026), 'name': 'Dar es Salaam (Potential)'},
    'LBV': {'coords': (0.4586, 9.4123), 'name': 'Libreville (Potential)'},
    'BKO': {'coords': (12.5335, -7.9499), 'name': 'Bamako (Potential)'},
    'BGF': {'coords': (4.3985, 18.5188), 'name': 'Bangui (Potential)'},
    'ADD': {'coords': (8.9779, 38.7993), 'name': 'Addis Ababa'},
    'PER': {'coords': (-31.9505, 115.8605), 'name': 'Perth'},
    'ALG': {'coords': (36.6910, 3.2158), 'name': 'Algiers'},
    'CMN': {'coords': (33.5731, -7.5898), 'name': 'Casablanca'},
    'ABJ': {'coords': (5.3549, -4.0083), 'name': 'Abidjan'},
    'LUN': {'coords': (-15.3875, 28.3228), 'name': 'Lusaka'},
    'TNR': {'coords': (-18.8792, 47.5079), 'name': 'Antananarivo'}

}


from math import radians, cos, sin, asin, sqrt, atan2

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a),sqrt(1-a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r





def calculate_direct_routes(airports, hub_code='NBJ'):
    """Calculate and sort direct routes from the hub based on distance."""
    direct_routes = []
    for code, info in airports.items():
        if code != hub_code:  # Exclude the hub itself
            distance_km = haversine(airports[hub_code]['coords'][1], airports[hub_code]['coords'][0], 
                                    info['coords'][1], info['coords'][0])
            direct_routes.append((code, distance_km))
    # Sort routes by distance
    direct_routes.sort(key=lambda x: x[1])
    return direct_routes




def calculate_total_distance_for_connecting_routes(airports, direct_routes):
    """Calculate total distance for connecting routes A to B via hub and sort them."""
    connecting_routes = []
    for origin_code, origin_distance in direct_routes:
        for destination_code, destination_distance in direct_routes:
            if origin_code != destination_code:  # Ensure different origin and destination
                # Total distance is the sum of both legs
                total_distance_km = origin_distance + destination_distance
                connecting_routes.append((origin_code, destination_code, total_distance_km))
    # Sort connecting routes by total distance
    connecting_routes.sort(key=lambda x: x[2])
    return connecting_routes

def print_routes(direct_routes, connecting_routes):
    """Print the most feasible direct and connecting routes."""
    print("Most Feasible Direct Routes (sorted by distance):")
    for code, distance in direct_routes:
        print(f"- NBJ to {code}: {distance:.2f} km")
    print("\nMost Feasible Connecting Routes (sorted by total distance):")
    for origin, destination, total_distance in connecting_routes:
        print(f"- {origin} to {destination} via NBJ: {total_distance:.2f} km")

# Assuming direct_routes calculation from previous code
direct_routes = calculate_direct_routes(airports)

# Calculate total distances for connecting routes
connecting_routes = calculate_total_distance_for_connecting_routes(airports, direct_routes)

# Print routes
#print_routes(direct_routes, connecting_routes)



def calculate_direct_routes_from_hub(airports, hub_code):
    """Calculate and sort direct routes from a given hub based on distance."""
    direct_routes = []
    for code, info in airports.items():
        if code != hub_code:  # Exclude the hub itself
            distance_km = haversine(airports[hub_code]['coords'][1], airports[hub_code]['coords'][0], 
                                    info['coords'][1], info['coords'][0])
            direct_routes.append((code, distance_km))
    # Sort routes by distance
    direct_routes.sort(key=lambda x: x[1])
    return direct_routes

def calculate_total_distance_for_connecting_routes_via_hubs(airports, hubs):
    """Calculate total distance for connecting routes via specified hubs and compare them."""
    connecting_routes = {}
    for hub_code in hubs:
        direct_routes = calculate_direct_routes_from_hub(airports, hub_code)
        for origin_code, origin_distance in direct_routes:
            for destination_code, destination_distance in direct_routes:
                if origin_code != destination_code:  # Ensure different origin and destination
                    # Total distance is the sum of both legs
                    total_distance_km = origin_distance + destination_distance
                    route_key = (origin_code, destination_code)
                    if route_key not in connecting_routes or total_distance_km < connecting_routes[route_key]['distance']:
                        # Update if this hub offers a shorter route or if route is new
                        connecting_routes[route_key] = {'hub': hub_code, 'distance': total_distance_km}
    return connecting_routes

def print_hub_benchmark(connecting_routes):
    """Print the best hub option for each connecting route based on total distance."""
    print("Best Hub Option for Connecting Routes (sorted by route):")
    for route_key, info in sorted(connecting_routes.items()):
        print(f"- {route_key[0]} to {route_key[1]} via {info['hub']}: {info['distance']:.2f} km")

# Define hubs for comparison
hubs = {'NBJ': {'coords': (-8.85837, 13.2312), 'name': 'Luanda'},
        'ADD': {'coords': (8.9779, 38.7993), 'name': 'Addis Ababa'},
        'DXB': {'coords': (25.2528, 55.3644), 'name': 'Dubai'},
        'DOH': {'coords': (25.2611, 51.5651), 'name': 'Doha'},
        'LIS': {'coords': (38.7704, -9.1291), 'name': 'Lisbon'},
        'JNB': {'coords': (-26.1392, 28.246), 'name': 'Johannesburg'}}

# Assuming 'airports' dictionary includes all relevant airports

# Calculate and compare connecting routes via hubs
connecting_routes_comparison = calculate_total_distance_for_connecting_routes_via_hubs(airports, hubs)

# Print the benchmark comparison
#print_hub_benchmark(connecting_routes_comparison)


def print_routes_optimized_for_hub(connecting_routes, target_hub='NBJ'):
    """Print connecting routes optimized for a specific hub."""
    print(f"Routes Optimized for {target_hub} Hub:")
    for route_key, info in sorted(connecting_routes.items()):
        if info['hub'] == target_hub:
            print(f"- {route_key[0]} to {route_key[1]} via {info['hub']}: {info['distance']:.2f} km")

# Assuming 'connecting_routes_comparison' contains the comparison result
# Print the routes optimized for NBJ
print_routes_optimized_for_hub(connecting_routes_comparison, 'NBJ')



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Function to plot optimized network for a hub
def plot_optimized_network_for_hub(airports, connecting_routes_comparison, target_hub='NBJ'):
    plt.figure(figsize=(15, 20))
    m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=70, llcrnrlon=-130, urcrnrlon=160, lat_ts=20, resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue')

    # Plot airport locations
    for code, info in airports.items():
        x, y = m(info['coords'][1], info['coords'][0])
        m.plot(x, y, marker='o', color='green' if code == target_hub else 'blue', markersize=10 if code == target_hub else 5)
        plt.text(x, y, code, fontsize=12)

    # Plot routes optimized for target_hub
    for route_key, info in connecting_routes_comparison.items():
        if info['hub'] == target_hub:
            origin = airports[route_key[0]]['coords']
            destination = airports[route_key[1]]['coords']
            ox, oy = m(origin[1], origin[0])
            dx, dy = m(destination[1], destination[0])
            m.drawgreatcircle(origin[1], origin[0], destination[1], destination[0], color='red', linewidth=2)

    plt.title(f'Network Optimized for {airports[target_hub]["name"]} Hub')
    plt.show()

# Assuming 'airports' dictionary includes NBJ and all relevant airports
# And 'connecting_routes_comparison' contains the hub benchmark comparison results

# Plot the network optimized for NBJ
plot_optimized_network_for_hub(airports, connecting_routes_comparison, 'NBJ')
