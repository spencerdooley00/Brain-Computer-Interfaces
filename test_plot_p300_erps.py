# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 12:43:40 2021

@author: Spencer Dooley and Kristina Sitcawich
"""

# Import functions used from other modules
from load_p300_data import load_train_eeg
from plot_p300_erps import get_events
from plot_p300_erps import epoch_data
from plot_p300_erps import get_erps 
from plot_p300_erps import plot_erps

#load the training eeg_data
eeg_time, eeg_data, rowcol_id, is_target = load_train_eeg(subject=3)

# get the event sample array and is_target_event array
event_sample, is_target_event=get_events (rowcol_id,is_target)

#epoch the eeg data and retreive array of erp times relative to event
eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample)

#get the target and non target erps
target_erp, nontarget_erp = get_erps(eeg_epochs, is_target_event)

#plot the erp data 
plot_erps(target_erp, nontarget_erp, erp_times)


