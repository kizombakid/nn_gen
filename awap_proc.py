from netCDF4 import Dataset
import numpy as np
import datetime
import sys
import matplotlib.pyplot as plt


def find_nearest_gridpoint(lat,lon,location):

    '''
    Purpose: find the nearest grid point to a given location
    Input parameters:
    lat - grid latitudes
    lon - grid longitudes
    location - requested [lat,long]
    Return:
    [i,j] grid index of requested lat long
    '''

    ipos = (np.abs(lon-location[1])).argmin()
    jpos = (np.abs(lat-location[0])).argmin()
    return ipos,jpos

lname = 'penrith_m5'
location = [-33.75, 150.69]

var = sys.argv[1]
year1 = 1911
year2 = 2018

date = []
vph = []
vpl_c = []
vpl_n = []
vpl_s = []
vpl_e = []
vpl_w = []


for year in range(year1,year2+1):
    ncfile = Dataset('/data/awap/'+var+'_'+str(year)+'.nc',"r",format="NETCDF4")
    time = ncfile.variables['time'][:]
    lat = ncfile.variables['lat'][:]
    lon = ncfile.variables['lon'][:]
    val = ncfile.variables[var][:]
    ncfile.close()

    # take out leap year
    print (year, len(time))
    if len(time) == 366:
        time = time[:-1]
        val = val[:-1,:,:]

    si = np.shape(val)

    ipos,jpos = find_nearest_gridpoint(lat,lon,location)

    dt0 = datetime.datetime(year, 1, 1, 0, 0)


    for it in range(0,len(time)):

        dt = dt0 + datetime.timedelta(days=it)
        dts = dt.strftime("%Y%m%d")

        date.append(int(dts))

        if it==0:
            mn=np.mean(val[it,jpos-5:jpos+6, ipos-5:ipos+6])
            print (mn,val[it,jpos,ipos-5])

        vph.append(val[it,jpos,ipos-5])
        vpl_c.append(np.mean(val[it,jpos-5:jpos+6, ipos-5:ipos+6]))
        vpl_n.append(np.mean(val[it,jpos-5+10:jpos+6+10, ipos-5:ipos+6]))
        vpl_s.append(np.mean(val[it,jpos-5-10:jpos+6-10, ipos-5:ipos+6]))
        vpl_e.append(np.mean(val[it,jpos-5:jpos+6, ipos-5+10:ipos+6+10]))
        vpl_w.append(np.mean(val[it,jpos-5:jpos+6, ipos-5-10:ipos+6-10]))

np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_date.npy',date)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_vph.npy',vph)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_vpl_c.npy',vpl_c)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_vpl_n.npy',vpl_n)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_vpl_s.npy',vpl_s)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_vpl_e.npy',vpl_e)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_vpl_w.npy',vpl_w)

# alse create anomalies relative to whole period
def create_anom(value):
    print (len(value))
    nyears = len(value)//365
    rvalue = np.reshape(value,(nyears,365))

    avalue = np.copy(rvalue)
    si = np.shape(rvalue)
    ny = si[0]
    nt = si[1]
    for n in range(0,nt):
        mean = np.mean(rvalue[:,n])
#        if n ==0 : print ('Elements ', rvalue[n,:])
        avalue[:,n] = rvalue[:,n] - mean

    return avalue.flatten()

aph = create_anom(vph)
apl_c = create_anom(vpl_c)
apl_n = create_anom(vpl_n)
apl_s = create_anom(vpl_s)
apl_e = create_anom(vpl_e)
apl_w = create_anom(vpl_w)

np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_aph.npy',aph)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_apl_c.npy',apl_c)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_apl_n.npy',apl_n)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_apl_s.npy',apl_s)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_apl_e.npy',apl_e)
np.save('/home/oscar/analyse/data/awap_series/'+lname+'_'+var+'_apl_w.npy',apl_w)

plt.figure()
plt.plot(vph[0:365*3])
plt.show()
plt.figure()
plt.plot(aph[0:365*3])
plt.show()

'''
#Checker for nyears=9
mn=0
for n in range(9):
    mn=mn+vph[n*365]
mn=mn/9
print ('mn',mn)
print (vph[0]-mn,vph[365]-mn,vph[365*2]-mn)
print (aph[0],aph[365],aph[365*2])
# Test
#var = np.load('/home/oscar/analyse/nn_batch/exps/e_lead1/CBA/dataA_vali_v.npy')
'''
