# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:01:26 2021

Lab 5: Spatial Components

Test script to call the following functions frfom remove_audvis_blinks.py
load_data
plot_components
get_sources
remove_sources
compare_reconstruction

@author: spencer, JJ
"""
import remove_audvis_blinks
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (14,8)


#%% Part 1: Load the Data
data_dir = ''
channels_to_plot = ['Fpz', 'Cz', 'Iz']
data = remove_audvis_blinks.load_data(data_dir, channels_to_plot)

# Zooming in on the raw data for electrodes Fpz, Cz, and Iz, it looks like a blink occured at around 14 seconds on 
# all three electrodes. I know this because you can see a clear peak at around 200 mV on electrode Fpz, and this is
# typical of ocular artifacts in EEG data. Electrode Fpz had the highest peak from the blink artifact, while electrodes Cz
# and Iz had much lower peaks at around 70 and 50 mV, respectively.

#%% Part 2: Plot the Components
components_to_plot = np.arange(0,10,1)

mixing_matrix = data['mixing_matrix']
channels = data['channels']

remove_audvis_blinks.plot_components(mixing_matrix, channels, components_to_plot)

# Of the 10 components pictured, it appears that components 0, 1, and 9 contain EOG artifacts based on the topomaps. You can clearly see
# on the topomap for these three components that there is high polarity around the electrodes placed near the subjects eyes, and we 
# would expect this from EOG artifacts.

#%% Part 3: Transform into Source Space
eeg = data['eeg']
unmixing_matrix = data['unmixing_matrix']
fs = data['fs']
sources_to_plot = [0,3,9]
source_activations = remove_audvis_blinks.get_sources(eeg, unmixing_matrix, fs, sources_to_plot)

#%% Part 4: Remove Artifact Components
cleaned_eeg = remove_audvis_blinks.remove_sources(source_activations, mixing_matrix, sources_to_remove= [0,3,9])
reconstructed_eeg = remove_audvis_blinks.remove_sources(source_activations, mixing_matrix, sources_to_remove =[])

#%% Part 5: Transform Back Into Electrode Space
remove_audvis_blinks.compare_reconstructions(eeg, cleaned_eeg, reconstructed_eeg, fs, channels, channels_to_plot)

# Describing our results: Looking at the resulting figure that compares the raw, reconstructed, and cleaned EEG data, we can clearly see that 
# our ICA artifact removal worked in removing the large voltage spikes that occured when the subject blinked. For example in channel Fpz, our cleaned EEG
# data remains at a voltage around 20 mV instead of shooting up to 200 mV at the 14 second mark like the raw EEG data does. Overall at a quick visual glance
# you can clearly see the cleaned EEG data more accuractly represents the expected brain response in teh subject. We know before plotting which electrodes will more 
# strongly show the effects of removing a particular independant component because we can see in our topographical map the components that are most affected by 
# blinking artifacts in the raw EEG data. We also know that electrodes in the frontal head area, especially electrodes closer to the eyes, are the electrodes usually most 
# effected by blink artifacts. Lateral eye movements can also affect recorded EEG data in the frontal area of the head, as well as show up as artifacts in the 
# electrodes close to a subjects temples. When we removed components where we did not think there were artifacts present, the cleaned eeg more closely matched the original and recontructed eeg, 
# where as when sources that had artifacts were removed, teh cleaned eeg was much different.

