import foursquare
import os
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Foursquare credentials')

parser.add_argument('-id', '--client_id', required=False)
parser.add_argument('-s', '--client_secret', required=False)

args = parser.parse_args()

data = pd.read_csv("data/merged_data_no_na.csv")
data = data[['id', 'atm_group', 'address', 'address_rus', 'lat', 'long', 'isTrain', 'target']]

client_id = args.client_id
client_secret = args.client_secret
experiment_name = 'terminal'
basepath = 'places-parser/data/%s' % experiment_name
clean_before_start = True

fq = foursquare.Foursquare(client_id=client_id, client_secret=client_secret)

# Airport

points = data[['id', 'lat', 'long']].copy()
venues_header_exists = not clean_before_start
requests_header_exists = not clean_before_start

if clean_before_start:
    if os.path.exists("%s/places-index.csv" % basepath):
        os.remove("%s/places-index.csv" % basepath)
    if os.path.exists("%s/reqs-index.csv" % basepath):
        os.remove("%s/reqs-index.csv" % basepath)

letsstart = True
request_index = []
for (key, point) in points.iterrows():
    if letsstart:
        response = search_venues(fq, point['lat'], point['long'])
        result = []
        for venueData in response['venues']:
            result.append(parse_venue(venueData))

        if len(result) > 0:
            venues_df = pd.DataFrame(result)
            venues_df['point_id'] = point['id']
            venues_df['request_lat'] = point['lat']
            venues_df['request_lon'] = point['long']
            venues_df.to_csv("%s/places-indexes.csv" % basepath, mode='a', header=(not venues_header_exists), index=False)
            venues_header_exists = True

        point['count'] = len(result)
        pd.DataFrame(point).T.to_csv("%s/reqs-indexes.csv" % basepath, mode='a',
                                     header=(not requests_header_exists), index=False)
        print(point['id'], point['count'])
        requests_header_exists = True

#Bus-station

experiment_name = 'busstation'
points = data[['id', 'lat', 'long']].copy()
venues_header_exists = not clean_before_start
requests_header_exists = not clean_before_start

if clean_before_start:
    if os.path.exists("%s/places-index.csv" % basepath):
        os.remove("%s/places-index.csv" % basepath)
    if os.path.exists("%s/reqs-index.csv" % basepath):
        os.remove("%s/reqs-index.csv" % basepath)

request_index = []
for (key, point) in points.iterrows():
    if letsstart:
        response = search_venues(fq,point['lat'], point['long'],categoryId = '4bf58dd8d48988d1fe931735')
        result = []
        for venueData in response['venues']:
            result.append(parse_venue(venueData))

        if len(result) > 0:
            venues_df = pd.DataFrame(result)
            venues_df['point_id'] = point['id']
            venues_df['request_lat'] = point['lat']
            venues_df['request_lon'] = point['long']
            venues_df.to_csv("%s/places-indexes.csv" % basepath, mode='a', header=(not venues_header_exists), index=False)
            venues_header_exists = True

        point['count'] = len(result)
        pd.DataFrame(point).T.to_csv("%s/reqs-indexes.csv" % basepath, mode='a',
                                     header=(not requests_header_exists), index=False)
        print(point['id'], point['count'])
        requests_header_exists = True

#Metro
experiment_name = 'metro'
points = data[['id', 'lat', 'long']].copy()
venues_header_exists = not clean_before_start
requests_header_exists = not clean_before_start

if clean_before_start:
    if os.path.exists("%s/places-index.csv" % basepath):
        os.remove("%s/places-index.csv" % basepath)
    if os.path.exists("%s/reqs-index.csv" % basepath):
        os.remove("%s/reqs-index.csv" % basepath)

request_index = []
for (key, point) in points.iterrows():
    if letsstart:
        response = search_venues(fq, point['lat'], point['long'], categoryId='4bf58dd8d48988d1fd931735')
        result = []
        for venueData in response['venues']:
            result.append(parse_venue(venueData))

        if len(result) > 0:
            venues_df = pd.DataFrame(result)
            venues_df['point_id'] = point['id']
            venues_df['request_lat'] = point['lat']
            venues_df['request_lon'] = point['long']
            venues_df.to_csv("%s/places-indexes.csv" % basepath, mode='a', header=(not venues_header_exists), index=False)
            venues_header_exists = True

        point['count'] = len(result)
        pd.DataFrame(point).T.to_csv("%s/reqs-indexes.csv" % basepath, mode='a',
                                     header=(not requests_header_exists), index=False)
        print(point['id'], point['count'])
        requests_header_exists = True

#Train station

experiment_name = 'trainstation'
points = data[['id', 'lat', 'long']].copy()
venues_header_exists = not clean_before_start
requests_header_exists = not clean_before_start

if clean_before_start:
    if os.path.exists("%s/places-index.csv" % basepath):
        os.remove("%s/places-index.csv" % basepath)
    if os.path.exists("%s/reqs-index.csv" % basepath):
        os.remove("%s/reqs-index.csv" % basepath)

request_index = []
for (key, point) in points.iterrows():
    if letsstart:
        response = search_venues(fq, point['lat'], point['long'], categoryId='4bf58dd8d48988d129951735')
        result = []
        for venueData in response['venues']:
            result.append(parse_venue(venueData))

        if len(result) > 0:
            venues_df = pd.DataFrame(result)
            venues_df['point_id'] = point['id']
            venues_df['request_lat'] = point['lat']
            venues_df['request_lon'] = point['long']
            venues_df.to_csv("%s/places-indexes.csv" % basepath, mode='a', header=(not venues_header_exists), index=False)
            venues_header_exists = True

        point['count'] = len(result)
        pd.DataFrame(point).T.to_csv("%s/reqs-indexes.csv" % basepath, mode='a',
                                     header=(not requests_header_exists), index=False)
        print(point['id'], point['count'])
        requests_header_exists = True

#Tram station

experiment_name = 'tram'
points = data[['id', 'lat', 'long']].copy()
venues_header_exists = not clean_before_start
requests_header_exists = not clean_before_start

if clean_before_start:
    if os.path.exists("%s/places-index.csv" % basepath):
        os.remove("%s/places-index.csv" % basepath)
    if os.path.exists("%s/reqs-index.csv" % basepath):
        os.remove("%s/reqs-index.csv" % basepath)

request_index = []
for (key, point) in points.iterrows():
    if letsstart:
        response = search_venues(fq, point['lat'], point['long'], categoryId='4bf58dd8d48988d1fc931735')
        result = []
        for venueData in response['venues']:
            result.append(parse_venue(venueData))

        if len(result) > 0:
            venues_df = pd.DataFrame(result)
            venues_df['point_id'] = point['id']
            venues_df['request_lat'] = point['lat']
            venues_df['request_lon'] = point['long']
            venues_df.to_csv("%s/places-indexes.csv" % basepath, mode='a', header=(not venues_header_exists),
                             index=False)
            venues_header_exists = True

        point['count'] = len(result)
        pd.DataFrame(point).T.to_csv("%s/reqs-indexes.csv" % basepath, mode='a',
                                     header=(not requests_header_exists), index=False)
        print(point['id'], point['count'])
        requests_header_exists = True

metro_df = pd.read_csv('places-parser/data/metro/places-indexes.csv')
airport = pd.read_csv('places-parser/data/terminal/places-indexes.csv')
train = pd.read_csv('places-parser/data/trainstation/places-indexes.csv')
bus = pd.read_csv('places-parser/data/busstation/places-indexes.csv')
tram = pd.read_csv('places-parser/data/tram/places-indexes.csv')
data_df = data.copy()
target = data[['target']]

metro_df = metro_df[['point_id', 'distance', 'primary_category']]
airport = airport[['point_id', 'distance', 'primary_category']]
train = train[['point_id', 'distance']]
bus = bus[['point_id', 'distance', 'primary_category']]
tram = tram[['point_id', 'distance', 'primary_category']]

metro_df['is_metro'] = np.where(metro_df.primary_category == 'Metro Station', 1, -1)
metro_df['metro_distance'] = np.where(metro_df.primary_category == 'Metro Station', metro_df.distance, -1)
metro_df = metro_df.rename(columns={'point_id':'id'})
metro_df = metro_df.drop_duplicates(keep='first', subset='id')
data_df = pd.merge(data, metro_df, how='left', on='id')
data_df = data_df.fillna(-1)

airport = airport[(airport.primary_category == 'Airport Terminal') | (airport.primary_category == 'Airport')]
airport = airport.rename(columns={'point_id':'id', 'distance':'air_distance', 'primary_category':'cat_airport'})
airport = airport[['id', 'air_distance']]
airport.id = airport.id.astype(str)
airport = airport.drop_duplicates(keep='first', subset='id')
data_df = pd.merge(data_df, airport, how='left', on='id')
data_df = data_df.fillna(-1)

train = train.rename(columns={'point_id':'id', 'distance':'train_distance'})
train.id = train.id.astype(str)
train = train.drop_duplicates(keep='first', subset='id')
data_df = pd.merge(data_df, train, how='left', on='id')
data_df = data_df.fillna(-1)

bus = bus[(bus.primary_category == 'Bus Line') | (bus.primary_category == 'Bus Stop')]
bus = bus.rename(columns={'point_id':'id', 'distance':'bus_distance', 'primary_category':'cat_bus'})
bus = bus[['id', 'bus_distance']]
bus.id = bus.id.astype(str)
bus = bus.drop_duplicates(keep='first', subset='id')
data_df = pd.merge(data_df, bus, how='left', on='id')
data_df = data_df.fillna(-1)

tram = tram[tram.primary_category == 'Tram Station']
tram = tram.rename(columns={'point_id':'id', 'distance':'tram_distance', 'primary_category':'cat_tram'})
tram = tram[['id', 'tram_distance']]
tram.id = tram.id.astype(str)
tram = tram.drop_duplicates(keep='first', subset='id')
data_df = pd.merge(data_df, tram, how='left', on='id')
data_df = data_df.fillna(-1)

data_df['metro_distance'] = data_df.distance.astype(float)
data_df['air_distance'] = data_df.metro_distance.astype(float)
data_df['train_distance'] = data_df.distance.astype(float)
data_df['bus_distance'] = data_df.metro_distance.astype(float)
data_df['tram_distance'] = data_df.metro_distance.astype(float)
data_df.id = data_df.id.astype(float)
data_df['target'] = target.target
data_df.to_csv('data/data_with_places.csv', index=False)