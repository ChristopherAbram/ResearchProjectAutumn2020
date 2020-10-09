from utils.location import GeoLocation


# list of strings
places = ['Hong Kong', 'Lagos, Nigeria']

geo = GeoLocation()
print("By place name")
for place in places:
    data = geo.get_coordinates(place)
    print(data)

print("All places at once")
data = geo.get_coordinates(places)
print(data)
