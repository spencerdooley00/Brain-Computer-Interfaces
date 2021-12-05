# -*- coding: utf-8 -*-
"""


test_load_p300_data.py

Contains code that is used to build functions in load_p300_data.py. Here we load the p300 data, 
extract the features we want, and plot the p300 data for subject 3

- Created 9/7/2021 by Spencer Dooley and Steve O'Driscoll


"""
#%%
# set up path to data and declare subject as a variable
data_directory = 'P300Data'
subject = 3
data_file = data_directory + f'/s{subject}.mat'

#%%
import numpy as np
import matplotlib.pyplot as plt
import loadmat

# load the training data for whatever subject we have subject set to
data = loadmat.loadmat(data_file)
train_data = data[f's{subject}']['train']

#%%
# Using slicing, extract the rows and assing them to arrays named accordingly
# We want eeg_time, eeg_data, the rowcol_id and the boolean array of is_target
eeg_time = train_data[0, :]
eeg_data = train_data[1:9, :].T
rowcol_id = train_data[9, :]
is_target = train_data[10, :]

#Set is_target to boolean array and rowcol_id to int
is_target = np.array(is_target, dtype=bool)
rowcol_id = np.array(rowcol_id, dtype=int)


#%%

ax1=plt.subplot(3, 1, 1)
plt.plot(eeg_time, rowcol_id)
#select only 5 seconds of data
plt.xlim((48,53))
plt.grid()
#add axis label
plt.ylabel("row/col ID")
#add title
plt.title(f'P300 Speller Subject {subject} Raw Data')

# second subplot
ax2=plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(eeg_time, is_target)
plt.xlim((48,53))
plt.grid()
plt.ylabel("Target ID")

#third subplot
ax3=plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(eeg_time, eeg_data)
plt.xlim((48,53))
plt.ylim((-25,25))
plt.grid()
#add x axis label
plt.xlabel("Time(s)")

plt.ylabel("Voltage(uV)")

plt.tight_layout()

#%%
# import the module created with load_p300_data.py
# call both functions
import load_p300_data
eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_train_eeg()
load_p300_data.plot_raw_eeg(subject, eeg_time, eeg_data, rowcol_id, is_target)
#%%
#call the load_and_plot_all function from load_p300_data
load_p300_data.load_and_plot_all([3,4,5,9])

#%%
# print the docstrings for each function in load_and_plot_all
print(load_p300_data.load_train_eeg.__doc__)
print(load_p300_data.plot_raw_eeg.__doc__)
print(load_p300_data.load_and_plot_all.__doc__)


#%%