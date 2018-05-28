#Make a running accumulated water year-to-date precipitation plot
#Plot inspiration and nicely formatted .csv GHCN-D output from Jared Rennie's GHCNpy repository
#Created by Massey Bartolini, 3/25/2018

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from scipy.ndimage.filters import gaussian_filter

stid='USW00014735' #KALB
if stid=='USW00014735':
	stname='Albany International Airport (KALB)'

#Read reformatted csv file
df=pd.read_csv(stid+'.csv',parse_dates=[[0,1,2]])
df=df.set_index('YYYY_MM_DD') #Set datetime as index

#Get starting and ending years in station record
syr=str(pd.to_datetime(df.index).year.values[0])
eyr=str(pd.to_datetime(df.index).year.values[-1])
title=stname+'\nAnnual Precipitation Time Series | Period of Record: '+syr+'-'+eyr #Plot title

#Unit conversions: Temps from C to F and precip from mm to in 
df['TMAX']=df['TMAX']*1.8+32.
df['TMIN']=df['TMIN']*1.8+32.
df.TMAX=np.around(df.TMAX,decimals=0).astype(int) #Round to nearest whole number before truncating to integers
df.TMIN=np.around(df.TMIN,decimals=0).astype(int)
df['PRCP']=np.around(df['PRCP']*0.03937,decimals=2)
df['SNOW']=np.around(df['SNOW']*0.03937,decimals=1)
df['SNWD']=np.around(df['SNWD']*0.03937,decimals=1)
df[df < -150.]=np.nan #Mark huge negative values as nan

#Get masks for every water year (Oct 1 to Sep 30) in dataframe
df['DOY']=df.index.dayofyear
idx=df.index[df.DOY==274]
print ('End of record for '+stid+': ')
print (df.tail())

water_yr_masks=[]
for i in range(len(idx)-1): #Find ranges of indices (masking) corresponding to each water year
	start_date=pd.to_datetime(idx[i])
	end_date=pd.to_datetime(idx[i+1])
	if i==0:
		mask=(df.index < start_date) #Beginning of record to first instance of Oct 1
	else:
		mask=(df.index >= start_date) & (df.index < end_date) #Oct 1 to Sep 30
	water_yr_masks.append(mask)

#Add current year's mask to list
water_yr_masks.append((df.index >= end_date)) #Oct 1 to present (could be partial year)

#Create figure
fig,ax=plt.subplots(figsize=(12,8))

#Loop through and count annual precip amounts
precip_acc_years=pd.DataFrame({}) #New dataframe for annual total precip 
precip_acc_timeseries=pd.DataFrame({}) #Each individual year's running summed precip, aligned to oct 1 (stripped of year info), used to compute mean/percentile stats
dfwatyr_all=[]
for i,mask in enumerate(water_yr_masks[1:]): #Skip first (partial) year
            dfwatyr=df.loc[mask] #Find water year
            dfwatyr=dfwatyr.fillna(value=0.0) #Fill missing data as zeros (won't hurt acc precip except for long periods of missing data)
            dfwatyr['PRCP_ACC']=dfwatyr.PRCP.cumsum() #Summation
            dfwatyr_all.append(dfwatyr)

            #Add cumulative precip total to summary dataframe
            precip_acc_years=precip_acc_years.append({'END_YR':int(dfwatyr.index.year[-1]),'PRCP_ACC':dfwatyr.PRCP_ACC.values[-1]},ignore_index=True)

            #Append cumulative precip series (full daily info) to dataframe if full year's worth of data exists
            precip_acc_yr=pd.Series(dfwatyr.PRCP_ACC.values,name='PRCP_ACC_'+str(dfwatyr.index.year[0])+'_'+str(dfwatyr.index.year[-1]))
            if (df.loc[mask].PRCP.count()>364) & (df.loc[mask].PRCP.count()<367):
                precip_acc_timeseries=precip_acc_timeseries.append(precip_acc_yr)
		
#Find top 5 rainiest and least rainy years to highlight these particular seasons in plot
precip_acc_years_sorted=precip_acc_years.sort_values('PRCP_ACC',ascending=False) #Sort from most to least rainy
print ('Rainiest years: ')
print (precip_acc_years_sorted.head())
if (df.index.month[-1] > 9) or (df.index.month[-1] < 10): #Don't include current year (incomplete) in top/bottom 5 stats
    precip_acc_years_sorted_dropcurryr=precip_acc_years_sorted[precip_acc_years_sorted.END_YR != float(eyr)]
else:
    precip_acc_years_sorted_dropcurryr=precip_acc_years_sorted
top5=list(precip_acc_years_sorted_dropcurryr.index[:5])
bot5=list(precip_acc_years_sorted_dropcurryr.index[-5:])
sorted_idx=list(precip_acc_years_sorted.index)
for i in sorted_idx: #Sorted, plotting sequentially from most to least rainy (nice order for legend)
	dfwatyr=dfwatyr_all[i]
	dates=[datetime(2017,10,1)+timedelta(days=i) for i in range(len(dfwatyr.PRCP_ACC.values))]
	pathfx=[PathEffects.withStroke(linewidth=3,foreground='w')]	
	top5colors=plt.cm.Spectral(np.linspace(0.7,1.0,5))[::-1]
	bot5colors=plt.cm.Spectral(np.linspace(0,0.3,5))[::-1]
	if i in top5: #Plot top5 with a different color and linewidth to highlight
		ix=top5.index(i)
		ci=top5colors[ix] #Pick color
		ax.plot(dates,dfwatyr.PRCP_ACC.values,color=ci,lw=2,zorder=3,label=str(dfwatyr.index.year[0])+'-'+str(dfwatyr.index.year[-1])+' ('+str(dfwatyr.PRCP_ACC.values[-1])+' in.)')
	elif i in bot5:	#Similar to top5, plot with different color/linewidth
		ix=bot5.index(i)
		ci=bot5colors[ix] #Pick color
		ax.plot(dates,dfwatyr.PRCP_ACC.values,color=ci,lw=2,zorder=3,label=str(dfwatyr.index.year[0])+'-'+str(dfwatyr.index.year[-1])+' ('+str(dfwatyr.PRCP_ACC.values[-1])+' in.)')
	elif i==len(dfwatyr_all)-1: #Plot current ytd rainfall in black
		ax.plot(dates,dfwatyr.PRCP_ACC.values,color='k',lw=3,zorder=4)
		ax.plot(dates[len(dfwatyr.PRCP_ACC.values)-1],dfwatyr.PRCP_ACC.values[-1],'*',color='g',markersize=15,mec='k',mew=1,zorder=5,label=str(dfwatyr.index.year[0])+'-'+str(dfwatyr.index.year[-1])+' ('+str(dfwatyr.PRCP_ACC.values[-1])+' in.)')
	else: #Plot all other years in light blue with thin line
		ax.plot(dates,dfwatyr.PRCP_ACC.values,color='yellowgreen',lw=0.75,zorder=2.5)

#plot smoothed accumulated rain quantiles (10-90 light gray shading, 25-75 darker gray shading, and single line for 50)
q90=precip_acc_timeseries.quantile(q=0.9).values[:-1]
q75=precip_acc_timeseries.quantile(q=0.75).values[:-1]
q50=precip_acc_timeseries.quantile(q=0.5).values[:-1]
q25=precip_acc_timeseries.quantile(q=0.25).values[:-1]
q10=precip_acc_timeseries.quantile(q=0.1).values[:-1]
dates=[datetime(2017,10,1)+timedelta(days=i) for i in range(len(q90))]
ax.fill_between(dates,gaussian_filter(q90,sigma=5),gaussian_filter(q10,sigma=5),color='0.7',alpha=0.3,label='10th-90th Percentile\n('+str(np.around(q10[-1],decimals=1))+'-'+str(np.around(q90[-1],decimals=1))+' in.)')
ax.fill_between(dates,gaussian_filter(q75,sigma=5),gaussian_filter(q25,sigma=5),color='0.5',alpha=0.3,label='25th-75th Percentile\n('+str(np.around(q25[-1],decimals=1))+'-'+str(np.around(q75[-1],decimals=1))+' in.)')
ax.plot(dates,gaussian_filter(q50,sigma=5),color='0.5',lw=2,label='50th Percentile ('+str(np.around(q50[-1],decimals=1))+' in.)',zorder=2.7)
	
#Plot credits
credits=True
if credits:
	pathfx=[PathEffects.withStroke(linewidth=3,foreground='w')]	
	xy_credit=(0.99,0.01)
	plt.annotate('Plot by Massey Bartolini', xy=xy_credit, xycoords='axes fraction', horizontalalignment='right',verticalalignment='bottom',fontsize=9,color='k',path_effects=pathfx,zorder=10)
	xy_credit=(0.01,0.01)
	plt.annotate('GHCN-D data courtesy of NOAA/NCEI', xy=xy_credit, xycoords='axes fraction', horizontalalignment='left',verticalalignment='bottom',fontsize=9,color='k',path_effects=pathfx,zorder=10)

#Legend, axis labels, etc
plt.legend(loc='upper left')
#Set custom x-axis date formatting
mons=mpl.dates.MonthLocator(range(1,13),bymonthday=1,interval=1)
monsfmt=mpl.dates.DateFormatter('1 %b')
ax.xaxis.set_major_locator(mons) #Label axis at every month
ax.xaxis.set_major_formatter(monsfmt)
plt.grid(linestyle='--')
#Set plot x-limits to water year (Oct - Sep)
ax.set_xlim(datetime(2017,9,27),datetime(2018,10,3))
plt.ylim([-3,precip_acc_years.PRCP_ACC.values.max()+3])
plt.xlabel('Date')
plt.ylabel('Accumulated Precipitation (in.)')
plt.suptitle(title,fontsize=12,fontweight='bold')
plt.subplots_adjust(top=0.93)
plt.savefig(stid.lower()+'_precip_acc_ytd.png',bbox_inches='tight')
plt.close()
