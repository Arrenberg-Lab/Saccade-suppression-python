# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:31:45 2020

@author: Ibrahim Alperen Tunc
"""

"""
-TODOs: 
    -implement saving procedure for the saccade data to prevent long time waiting. Do this the latest, when you can reliably extract the saccades
    +set a saccade magnitude threshold, and discard all eye movements not being unilateral. Then you need to do further stuff for error estimation
    DONE in zf_helper_funcs
    -extract saccade onsets etc for averaging over multiple trials
    -take median average for the data smoothing (running median average)
    -use different kernels, maybe not considering past etc
    -plot the distribution of saccade sizes.
    -your noise filtering approach might be very harsh on the small saccades, as measurement noise might be additive, and your noise threshold is
    relative to saccade size.
"""