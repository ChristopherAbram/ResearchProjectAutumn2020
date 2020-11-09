from humset.utils.location import GeoLocation


# list of strings
places = ['New York', 'Hong Kong', 'Warszawa', 'Berlin']

geo = GeoLocation()
print("By place name")
for place in places:
    data = geo.get_coordinates(place)
    print(data)

print("All places at once")
data = geo.get_coordinates(places)
print(data)
