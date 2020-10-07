from geopy.geocoders import Nominatim
import time

app = Nominatim(user_agent='foo')

# we could pass it the ROIs as arguments, or have it read from a file

# list of strings
region_of_interest = ['Hong Kong', 'Lagos, Nigeria']

data = dict()

for roi in region_of_interest:
    location = app.geocode(roi).raw

    data[roi] = {   'latitude': location['lat'],\
                    'longitude': location['lon'],\
                    'box': location['boundingbox'] }

print(data)
