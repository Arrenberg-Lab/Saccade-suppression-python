# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:31:45 2020

@author: Ibrahim Alperen Tunc
"""

"""
-TODOs: 
    -implement saving procedure for the saccade data to prevent long time waiting. Do this the latest, when you can reliably extract the saccades
    .set a saccade magnitude threshold, and discard all eye movements not being unilateral. Then you need to do further stuff for error estimation
    DONE in zf_helper_funcs
    .extract saccade onsets etc for averaging over multiple trials
    .take median average for the data smoothing (running median average)
    -use different kernels, maybe not considering past etc
    .plot the distribution of saccade sizes.
    .your noise filtering approach might be very harsh on the small saccades, as measurement noise might be additive, and your noise threshold is
    relative to saccade size.
    
    07.01: .Create a function for generation of grating image -> spatial frequency dependent DONE
           .Improve Gabor filter: 
               !Size depends on screen resolution to express in angles DONE
               !Circular Gabor filters DONE
           -Implement population decoder for location, size and spatial frequency.
           => Problem: same grating will probably invoke same activity in different sized filters. Maybe ML decoding a 
           better approach. 
           
   19.01: .Update the model: choose 200 filters, randomize the parameters for each filter, tile the visual image for RF
          centers but jiiter the center of each neuron (Gabor RF 200 180*360 images). DONE
          
          .For now crop the receptive fields which are on the edge and not fitting into the image field. You need to
          do some geographic spherical transformations in the future. Geographic transformations pending.
          
          .Correlate image with the filters, then get the activity by summing over the filter array. DONE
          
          .Negative values: just take absolute as we are interested in relevance. DONE
          
          .We are interested in shift, so you need to determine the center of the stimulus within the visual field.
          For that purpose get the neuronal activity (1*n array) and dot product (weighted sum) it with the receptive 
          field centers of each neuron (2*n array x and y coordinates). This will give you the center of the stimulus 
          location, then do it for the shifted image (1, 5, 10, 20, 30° image shifts) and the degree of shift is the 
          inferred stimulus center differences between the images. DONE
          
          .Use a simpler stimulus (big white dot). DONE
          
          +When this works, explore the model parameters: generate different sets of parameters (fake species, maybe
          check out the literature for some reference point) and see what is going on.
          
          .After the point above, check the inter-individual variability. Generate a violin plot for 100-10k
          simulations  DONE
          
          .Violin plot also for different neuron numbers  DONE
          
          +Also try different types of stimuli => Naturalistic etc.
          
          +For later: check size-sf relationship in the Gabor filters. Look up the literature.
          
          .Increase the shift size, decrease the stimulus size (smaller than the RFs) DONE
          
          . RF size set: 2 to 100 degrees with logarithmic sampling (do for now 10 to 100 degrees since rsl) DONE
          
          +Spatial frequency values: 1) For zebrafish wait for Giulia response, 
                                     2) For macaque check the literature
          
          +For the filter parameters (size and spatial frequency) find the value distribution and sample from the
          values using those distributions
          
          +Ari's suggestion of the hybrid model: Implement a shift decoder based on phase difference.
          
   26.01: .Violin plots: Plot 1st and 99th quartiles, stuff outside there is outliers DONE
          
          .Normed error is relative, the other one is absolute. DONE
          
          +nfilters simulation, report nans as zero shift. Discuss tomorrow since nans are about location not
          about shifts.
          
          +phase shift decoder: Try to do an image reconstruction decoder, leave out the phase shift story, instead 
          weigh each filter with their respective activity and superimpose on each other to get the decoded image for
          a given filter population. SOMEHOW NOT REALLY WORKING
              
          .presentation: discuss tomorrow. rough draft until saturday morning (friday night is yours ;))
          
          .RERUN the simulation differing in nfilters since you dumbo saved the values wrong UGHHH DUMBO!
          
          +Thus generate also correct violinplots. Just go through them to save them.
                    
          +Run a new simulation with single image where the stimulus location also randomly jitters as much as the 
          average distance between filters (stimulus in the center of the visual field, if distance between filters is
          on average 60°, then the stimulus location jitters around +-30° in azimuth and elevation). Respective 
          violinplots. Simulation in progress.
    
          +First try increasing the filter number in the image reconstruction model (2000 or 5000), also considering 
          different orientations for the Gabor filter.
    
          +If this does not work out, try the model Ari suggested: Generate the filter population as in the shift 
          decoder model with random RF center locations and random RF parameters, leaving out the phase parameter. 
          Then for a given filter generate multiple copies of filters at the same location with the same parameters 
          only differing in phase. In other words generate different filter phase maps for a given filter set.
          Acquire the joint activity of all filter maps for a given stimulus, decode the stimulus shift also using 
          the activity difference between filters of different phase.
          
          +Yet for now I am really not sure how to approach for the decoding algorithm, one idea I have now is to 
          weight the activity of each filter in a map to get the location read-out within each map, then somehow 
          combine readouts of all filter maps considering the phase difference. Another idea would be comparing the 
          activity levels of filters differing only in phase. Of course I can also simply weight all the filter 
          center locations with the respective activity level regardless of the filter maps.
          
          +Improve lab meeting presentation until Monday morning.
"""