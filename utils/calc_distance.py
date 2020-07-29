from math import radians, cos, sin, asin, sqrt, atan2
import math


def distance(lat1, lat2, lon1, lon2): 
      
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
       
    # Haversine formula  
    dlon = lon2 - lon1  
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
#    c = 2 * atan2(sqrt(a), sqrt(1-a))  
    c = 2*asin(sqrt(a))
     
    # Radius of earth in kilometers. Use 3956 for miles 
    r = 6371
       
    # calculate the result 
    return(c * r) 

  
def gravity_law_commute_dist(lat1,lon1,pop1,lat2,lon2,pop2,r):
  
  d = int(distance(lat1,lat2,lon1,lon2))
  C=1
  
  if d <= 300:
    alpha = 0.46
    beta = 0.64    
    w = C*(pop1** alpha) * (pop2**beta)/(math.exp(d/r))
    
  elif d<=700:
    alpha=0.35
    beta=0.37
    w=C*(pop1** alpha) * (pop2**beta)/(math.exp(d/r))
  else:
    w=0    
  return w