from geopy.geocoders import Nominatim


class GeoLocation:

    def __init__(self, agent='humset'):
        self.__app = Nominatim(user_agent=agent)


    def get_coordinates(self, name):
        if type(name) is str:
            l = self.__app.geocode(name).raw
            return {   
                'lat': float(l['lat']), \
                'lon': float(l['lon']), \
                'bbox': l['boundingbox']
            }
        elif type(name) is list:
            data = dict()
            for roi in name:
                data[roi] = self.get_coordinates(roi)
            return data
