'''
Contains useful methods for getting elevation data from the USGS SRTM1 dataset.
A lot of code is borrowed and inspired from DEM-Tools at https://github.com/migurski/DEM-Tools

By Nick Crews and Harrison Selle
3/9/16
'''


import numpy as np
import math
import struct

from StringIO import StringIO
from zipfile import ZipFile
from httplib import HTTPConnection
from urlparse import urlparse


previously_loaded = {}


def get_data(bounds):
    """ Return a numpy array of elevation data for the real world area contained in bounds"""


    # get a list of all the tiles that we need. 
    # We only need one tile if we're in the center of it, but we might need multiple if we're near an edge or corner
    lonlats = list(quads(*bounds))
    # get a list of all the data arrays for these tiles
    arrays = [data_from_tile(lat, lon) for (lon, lat) in lonlats]
    # if we need to, merge the tiles data together into one big array
    complete_array = merge_arrays(arrays, lonlats)


    # now we need to get the right section out of this tile
    x1, y1, x2, y2 = get_slice(bounds, lonlats[0])
    return complete_array[x1:x2, y1:y2]
   
def data_from_tile(lat, lon):
    '''Get all the data from a certain tile'''
    # this will be the filename on the website and inside the downloaded zip file
    filename = 'N%02dW%03d.hgt' % (abs(lat), abs(lon))

    # check to see if we've cached this map previosuly
    if filename in previously_loaded:
        return previously_loaded[filename]

    # otherwise, we need to download
    # get the region that this tile will live in
    reg = region(lat, lon)
    zipdata = get_zip_data(reg, filename)

    # make a ZipFile object from the downloaded file and unzip the contents
    zipfile = ZipFile(StringIO(zipdata))
    rawdata = zipfile.open(filename).read()

    # make a numpy array out of the raw binary data
    # how many samples in one dimension
    dim = 3601
    format = ">%dH" % dim**2
    data = struct.unpack(format, rawdata)
    arr = np.array(data).reshape((dim, dim))
    # cache this map
    previously_loaded[filename] = arr
    return arr
  
def get_zip_data(region, filename):
    '''Given the region name and file name, download that zip file from the USGS website and return the raw text'''

    # construct the url of the file
    fmt = 'http://dds.cr.usgs.gov/srtm/version2_1/SRTM1/Region_%02d/%s.zip'
    url = fmt % (region, filename)
    # print 'Getting data from ' + url + ' in elevation.data_from_tile()'

    # get the data
    s, host, path, p, q, f = urlparse(url)
    conn = HTTPConnection(host, 80)
    conn.request('GET', path)
    resp = conn.getresponse()

    if resp.status == 404:
        # we're probably outside the coverage area
        print 'problem opening url: %s' % url
        return None

    # otherwise, success
    return resp.read()
    
def merge_arrays(arrays, lonlats):
    '''Given a list of numpy arrays of elevation data, and a corresponsing list of their longitudes and latitudes, merge the data into one big array. Arrays and lonlats should be ordered sout->north, then west->east'''
    if len(arrays) == 1:
        # We're in the center of a tile, there is only one array
        complete_array =  arrays[0]
    elif len(arrays) == 2:
        # we're at an edge
        lon1, lat1 = lonlats[0]
        lon2, lat2 = lonlats[1]
        # is this a vertical or horizontal stack?
        if lon1 < lon2:
            # its a horizontal stack
            complete_array = np.concatenate((arrays[0], arrays[1]), axis=1)
        else:
            # its a vertical stack
            complete_array = np.concatenate((arrays[1], arrays[0]), axis=0)

    else:
        # we must have hit a corner, and have 4 arrays
        lower_array = np.concatenate((arrays[0], arrays[1]), axis=1)
        upper_array = np.concatenate((arrays[2], arrays[3]), axis=1)
        complete_array = np.concatenate((upper_array, lower_array), axis=0)
    return complete_array

def get_slice(bounds, array_location):
    '''Given the bounds of data we want (in coordinates), and the location of the NW corner of the array, return the right indexes to slice into the array'''
    minlon, minlat, maxlon, maxlat = bounds
    arr_min_lon, arr_min_lat = array_location
    # the dimensions of one tile
    dim = 3601
    
    x1 = int(round((minlon - arr_min_lon) * dim))
    x2 = int(round((maxlon - arr_min_lon) * dim))
    y1 = int(round((minlat - arr_min_lat) * dim))
    y2 = int(round((maxlat - arr_min_lat) * dim))

    # make sure we get at least one data point, in case x1==x2 or y1==y2
    x2 = max(x2, x1+1)
    y2 = max(y2, y1+1)

    return (x1, y1, x2, y2)



def region(lat, lon):
    """ Return the SRTM1 region number of a given lat, lon.
    
        Map of regions:
        http://dds.cr.usgs.gov/srtm/version2_1/SRTM1/Region_definition.jpg
    """
    if 38 <= lat and lat < 50 and -125 <= lon and lon < -111:
        return 1
    
    if 38 <= lat and lat < 50 and -111 <= lon and lon < -97:
        return 2
    
    if 38 <= lat and lat < 50 and -97 <= lon and lon < -83:
        return 3
    
    if 28 <= lat and lat < 38 and -123 <= lon and lon < -100:
        return 4
    
    if 25 <= lat and lat < 38 and -100 <= lon and lon < -83:
        return 5
    
    if 17 <= lat and lat < 48 and -83 <= lon and lon < -64:
        return 6
    
    if -15 <= lat and lat < 60 and ((172 <= lon and lon < 180) or (-180 <= lon and lon < -129)):
        return 7
    
    raise ValueError('Unknown location: %s, %s' % (lat, lon))

def quads(minlon, minlat, maxlon, maxlat):
    """ Generate a list of southwest (lon, lat) for 1-degree quads of SRTM1 data. Ordered south->north, then west->east
    """
    lon = math.floor(minlon)
    while lon <= maxlon:

        lat = math.floor(minlat)
        while lat <= maxlat:
    
            yield lon, lat 
    
            lat += 1

        lon += 1

    




