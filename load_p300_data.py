# -*- coding: utf-8 -*-
"""


load_p300_data.py

Contains functions that load the p300 data, extracts the features we want, and plots the p300 data

- Created 9/7/2021 by Spencer Dooley and Steve O'Driscoll


"""
import numpy as np
import matplotlib.pyplot as plt
import loadmat
def load_train_eeg(subject=3, data_directory='C:/Users/spenc/Documents/UVM/Fall 2021/BME 296/BME296Git/P300Data'):
    '''
    

    Parameters
    ----------
    subject : int
        Integer that corresponds to the subject we are interested in. The default is 3.
    data_directory : str
        String that holds the path to the location of the data. The default is 'C:/Users/spenc/Documents/UVM/Fall 2021/BME 296/BME296Git/P300Data'.

    Returns
    -------
    eeg_time : array
        Array of time of each sample taken (in seconds) .
    eeg_data : array
        Array of EEG data.
    rowcol_id : integer array
        integer array that shows the event type.
    is_target : boolean array
        boolean array representing if the current row/col flashed includes the target letter or not.


    '''
    # set up path to data and declare subject as a variable
    data_directory = data_directory
    subject = subject
    data_file = data_directory + f'/s{subject}.mat'
    

    # load the training data for whatever subject we have subject set to
    data = loadmat.loadmat(data_file)
    train_data = data[f's{subject}']['train']
    
    # Using slicing, extract the rows and assing them to arrays named accordingly
    # We want eeg_time, eeg_data, the rowcol_id and the boolean array of is_target
    eeg_time = train_data[0, :]
    eeg_data = train_data[1:9, :].T
    rowcol_id = train_data[9, :]
    is_target = train_data[10, :]
    
    #Set is_target to boolean array and rowcol_id to int
    is_target = np.array(is_target, dtype=bool)
    rowcol_id = np.array(rowcol_id, dtype=int)
    

    #return eeg_time, eeg_data, rowcol_id, is_target
    return eeg_time, eeg_data, rowcol_id, is_target

def plot_raw_eeg(subject, eeg_time, eeg_data, rowcol_id, is_target):
    '''
    

    Parameters
    ----------
    subject : int
        Integer that corresponds to the subject we are interested in.
    eeg_time : array
        Array of time of each sample taken (in seconds) .
    eeg_data : array
        Array of EEG data.
    rowcol_id : integer array
        integer array that shows the event type.
    is_target : boolean array
        boolean array representing if the current row/col flashed includes the target letter or not.

    Returns
    -------
    None.

    '''
    #use subplots 
    # first subplot
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
    # save figure to png file in same folder
    plt.savefig(f'P300_S{subject}_training_rawdata.png')

    
def load_and_plot_all(subjects):
    '''
    

    Parameters
    ----------
    subjects : int
        Integer that corresponds to the subject we are interested in.

    Returns
    -------
    None.

    '''
    # for each subject in range from the first subject we specify to the last one we specify...
    for subject in range(len(subjects)):
        # call load_train_eeg
        eeg_time, eeg_data, rowcol_id, is_target = load_train_eeg(subject = subjects[subject])
        #call plot_raw_eeg
        plot_raw_eeg(subjects[subject], eeg_time, eeg_data, rowcol_id, is_target)
        #clear plot so we can plot new subjects data
        plt.clf()

