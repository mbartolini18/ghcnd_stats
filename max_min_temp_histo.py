#Plot max/min temperature histograms from parsed GHCN-D station data
#Nicely formatted .csv GHCN-D output from Jared Rennie's GHCNpy repository
#Plot inspiration from Joe Zagrodnik (https://twitter.com/joejoezz/status/931591823910248448)
#Created by Massey Bartolini, 3/25/2018

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import cmocean
from datetime import datetime, timedelta

stnid='KALB'
if stnid=='KALB':
        stid='USW00014735' #KALB
        d1high,d1low=63,43 #KALB
        d2high,d2low=73,40 #KALB
        d1dt,d2dt=datetime(2018,2,20),datetime(2018,2,21)

days_anno=True #Annotate observed high/low temps

df=pd.read_csv(stid+'.csv') #Read reformatted GHCN-D data
syr=str(df.YYYY.values[0])
eyr=str(df.YYYY.values[-1])
if stnid=='KALB':
	title='Albany International Airport (KALB)\nWinter (DJF) High/Low Temperature Histogram | Period of Record: '+syr+'-'+eyr

df=df.loc[df.MM.isin([12,1,2])] #Subset df for winter months (Dec-Feb)
imgname=stid.lower()+'_temp_histo_djf.png'
print ('Top 20 warmest days for '+stid+': ')
print ((df.sort_values('TMAX',ascending=False).head(n=20))) #Print data from top 20 warmest high temp days

#Unit conversions: temps from C to F (mask missing values)
tmax=np.ma.masked_less(df.TMAX.values*1.8+32,-100.)
tmin=np.ma.masked_less(df.TMIN.values*1.8+32,-100.)
#print (tmax.max(),tmax.min()) #Max and min temps in record
#print (tmin.max(),tmin.min())

#Create figure for 2D histogram plot with 1D histogram side panels
fig=plt.figure(figsize=(9,8))
gs=mpl.gridspec.GridSpec(3,3) #Grid formatting with unequal subplot sizes

#Primary 2D histogram plot
ax=plt.subplot(gs[1:,:-1])
bins=np.arange(int(tmin.min())+0.5,int(tmax.max())+1+0.5,1) #Histogram bins every deg F, centered on midpoint
cmap=cmocean.cm.thermal
colhist=0.5 #Color (sampled from cmap) for side 1D histograms
counts,xedges,yedges,_=plt.hist2d(tmin,tmax,bins=bins,cmin=1,cmap=cmap)
if days_anno: #Annotate observed extrema
	ax.plot(d1low,d1high,'*',color='orange',markersize=14,mec='k',mew=1,label='Observed: {dt}'.format(dt=d1dt.strftime('%d %b %Y')))
	ax.plot(d2low,d2high,'*',color='red',markersize=14,mec='k',mew=1,label='Observed: {dt}'.format(dt=d2dt.strftime('%d %b %Y')))
	plt.legend(loc='lower left')

#Plot 1:1 tmax/tmin line for reference
plt.plot(bins,bins,color='k')
plt.grid(linestyle='--')
#Set axis limits, labels, credits
plt.xlim([tmin.min()-1,tmin.max()+1])
plt.ylim([tmax.min()-1,tmax.max()+1])
plt.xlabel('Low Temperature ($^\circ$F)')
plt.ylabel('High Temperature ($^\circ$F)')
credits=True
if credits: #Annotate plot author/data credits
	pathfx=[PathEffects.withStroke(linewidth=3,foreground='w')]	
	xy_credit=(0.99,0.01)
	plt.annotate('Plot by Massey Bartolini', xy=xy_credit, xycoords='axes fraction', horizontalalignment='right',verticalalignment='bottom',fontsize=9,color='k',path_effects=pathfx,zorder=10)
	xy_credit=(0.01,0.99)
	plt.annotate('GHCN-D data courtesy of NOAA/NCEI', xy=xy_credit, xycoords='axes fraction', horizontalalignment='left',verticalalignment='top',fontsize=9,color='k',path_effects=pathfx,zorder=10)

#Make low temp 1D histogram (side panel above 2D histogram)
ax=plt.subplot(gs[:1,:2])
bins=np.arange(int(tmin.min())+0.5,int(tmin.max())+1+0.5,1)
plt.hist(tmin,bins=bins,color=cmap([colhist]))
plt.xlim([tmin.min()-1,tmin.max()+1])
plt.ylabel('Counts')
ax.set_xticklabels([])
plt.grid(linestyle='--')
if days_anno:
	plt.axvline(d1low,color='orange',ls='--')
	plt.axvline(d2low,color='red',ls='--')

#Make high temp 1D histogram (side panel to the right of 2D histogram)
ax=plt.subplot(gs[1:,-1])
bins=np.arange(int(tmax.min())+0.5,int(tmax.max())+1+0.5,1)
plt.hist(tmax,bins=bins,color=cmap([colhist]),orientation='horizontal')
plt.ylim([tmax.min()-1,tmax.max()+1])
plt.xlabel('Counts')
ax.set_yticklabels([])
plt.grid(linestyle='--')
if days_anno:
	plt.axhline(d1high,color='orange',ls='--')
	plt.axhline(d2high,color='red',ls='--')

#Overall plot title
plt.suptitle(title,fontweight='bold') 

#Inset colorbar for 2D histogram plot
cax=fig.add_axes([0.58,0.15,0.03,0.24])
plt.colorbar(cax=cax,label='Counts')
cax.yaxis.set_ticks_position('left')
cax.yaxis.set_label_position('left')

plt.subplots_adjust(top=0.92)
plt.savefig(imgname)
