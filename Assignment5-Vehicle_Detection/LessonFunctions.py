import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
class LF:
    
    frame_tracking = 0
    single_frame_boxes = [] # Box list for a single frame
    frame_seq_boxes = [] # boxes from previous 'n' frames
    y_start = [450, 400, 400, 380]
    y_stop = [642, 592, 560, 508]
    img_scale = [2.5, 2, 1.5, 1]
    def read_image(img):
        
        image = cv2.imread(img)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    def imageCc(img, color_space='RGB'):
                
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)
        
        # Return the converted image
        return feature_image

    
    
    def c_2RGB(image, color_space):
        '''Convert the image back to RGB'''
            
        # Convert the image to RGB from the its original colorspace
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_LUV2RGB)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
        else: feature_image = np.copy(image)
        
        # Return the RGB image
        return feature_image
    
    def convert_color(img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    
    
    def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

        # Make a copy of the image
        draw_img = np.copy(img)
        
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
        
        # Return the image copy with boxes drawn
        return draw_img
    
    
    
    def color_hist(img, nbins=32, bins_range=(0, 255)):
        channel1_hist = []
        channel2_hist = []
        channel3_hist = []
        #print('img shape')
        #print(img.shape)
        # Compute the histogram of the RGB channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)[0]
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)[0]
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)[0]
        #print('ch shape')
        #print(channel1_hist.shape)
        # Stack the histograms into a single feature vector
        hist_features = np.hstack((channel1_hist, channel2_hist, channel3_hist))
        #print('hist feature shape')
        #print(hist_features.shape)
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    
    
    def bin_spatial(image, size=(16, 16)):
        
        features = cv2.resize(image, size).ravel()
        return features
    
    
    def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
                
        if vis == True:
            features, hog_image = hog(img,
                                      orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      block_norm = 'L2-Hys',
                                      transform_sqrt=True,
                                      visualise=vis,
                                      feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img,
                           orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           block_norm = 'L2-Hys',
                           transform_sqrt=True,
                           visualise=vis,
                           feature_vector=feature_vec)
            return features
    
    def combFeatures(feature_image, color_space, spatial_feat, hist_feat, hog_feat, hist_bins, orient,
                          pix_per_cell, cell_per_block, hog_channel, spatial_size):
        
        # list to hold the image features
        file_features = []
        
        # Get the spatial features
        if spatial_feat == True:
            spatial_features = LF.bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        
        # Get the histogram features
        if hist_feat == True:
            hist_features = LF.color_hist(feature_image)
            file_features.append(hist_features)
        
        # Get the hog features
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                hog_feat1 = LF.get_hog_features(feature_image[:,:,0],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True)
                hog_feat2 = LF.get_hog_features(feature_image[:,:,1],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True)
                hog_feat3 = LF.get_hog_features(feature_image[:,:,2],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True)
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                feature_image = LF.imageCc(feature_image, color_space=color_space)
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY)
                hog_features = LF.get_hog_features(feature_image[:,:], orient,
                                                    pix_per_cell, cell_per_block, vis=False,
                                                    feature_vec=True)
            file_features.append(hog_features)
        
        # Return the features as a list
        return file_features
    
    
    def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                         hist_bins=32, orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                         spatial_feat=True, hist_feat=True, hog_feat=True):
             
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = LF.read_image(file)
            
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                
                feature_image = np.copy(image)  
            
            # Get the image features and append them to the list
            file_features = LF.combFeatures(feature_image, color_space, spatial_feat, hist_feat,
                                                  hog_feat,hist_bins, orient, pix_per_cell,
                                                  cell_per_block, hog_channel, spatial_size)
            features.append(np.concatenate(file_features))
        
        # Return the features as a list
        return features
    
    
    def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop = list(x_start_stop)
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = int(img.shape[1])
        if y_start_stop[0] == None:
            y_start_stop = list(y_start_stop)
            y_start_stop[0] = int(img.shape[0]/2)
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    
    
    def find_cars(img, ystart, ystop, scale, svc, X_scaler, hog_channel,
                  orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                  xstart=0, xstop=1280, color_space='RGB'):
        
        draw_img = np.copy(img)
        box_list = []

        # Crop the image to the prefered search area
        img_tosearch = img[ystart:ystop,xstart:xstop,:]
        # ctrans_tosearch = LF.c_2RGB(img_tosearch, color_space)
        ctrans_tosearch = LF.imageCc(img_tosearch, color_space=color_space)
        
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # Get the hog channel depending on selection
        if hog_channel == 0 or hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:,:,0]
        if hog_channel == 1 or hog_channel == 'ALL':
            ch2 = ctrans_tosearch[:,:,1]
        if hog_channel == 2 or hog_channel == 'ALL':
            ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 4  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image and for the selected channel(s)
        if hog_channel == 0 or hog_channel == 'ALL':
            hog1 = LF.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel == 1 or hog_channel == 'ALL':
            hog2 = LF.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if hog_channel == 2 or hog_channel == 'ALL':
            hog3 = LF.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                
                # Extract HOG for this patch
                if hog_channel == 0 or hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_single_channel = hog_feat1
                if hog_channel == 1 or hog_channel == 'ALL':
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_single_channel = hog_feat2
                if hog_channel == 2 or hog_channel == 'ALL':
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_single_channel = hog_feat3

                # Get the single channel or multi-channel hog features
                if hog_channel == 'ALL':
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_single_channel
                
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                #print(subimg.shape)
                # Get color features
                spatial_features = LF.bin_spatial(subimg, size=spatial_size)
                hist_features = LF.color_hist(subimg, nbins=hist_bins)
                
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)+xstart
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)

                    # Draw the box on the image
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(255,0,0), 4)

                    # Extract the bbox coordinates
                    box = (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)
                    box_list.append(box)

        # Return the image with the vehicle detection overlay
        return draw_img, box_list
    
    ## from lectures
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        
        # Return thresholded map
        return heatmap
    
    ## from lectures
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        # Return updated heatmap
        return heatmap
    
    ## from lectures
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        
        # Return the image
        return img
    ## single image vehicle detection
    def process_image_VD(svc, X_scaler):
        
        spatial_size = (32,32)
        histbin = 32
        cell_per_block = 2
        pixels_per_cel = 8
        orient = 9
        color_space = 'HLS'
        for img in glob.glob('./test_images/test*.png'):
            image = LF.read_image(img)
            
            # Create an empty heat map to draw on
            heat = np.zeros_like(image[:,:,0]).astype(np.float)

            # Get the box list from using the hog sub sampling technique
            out_img, range_3_box_list = LF.find_cars(image,
                                                 LF.y_start[3],
                                                 LF.y_stop[3],
                                                 LF.img_scale[3],
                                                 svc, X_scaler,
                                                 'ALL',
                                                 orient,
                                                 pixels_per_cel,
                                                 cell_per_block,
                                                 spatial_size,
                                                 histbin,
                                                color_space=color_space)
            
            out_img, range_2_box_list = LF.find_cars(image,
                                                     LF.y_start[2],
                                                     LF.y_stop[2],
                                                     LF.img_scale[2],
                                                  svc, X_scaler,
                                                  'ALL',
                                                  orient,
                                                  pixels_per_cel,
                                                  cell_per_block,
                                                  spatial_size,
                                                  histbin,
                                                  color_space=color_space)
            
            out_img, range_1_box_list = LF.find_cars(image,
                                              LF.y_start[1],
                                              LF.y_stop[1],
                                              LF.img_scale[1],
                                                   svc, X_scaler,
                                                   'ALL',
                                                   orient,
                                                   pixels_per_cel,
                                                   cell_per_block,
                                                   spatial_size,
                                                   histbin,
                                                   color_space=color_space)

            # combine box list from all ranges
            box_list = range_3_box_list + range_2_box_list + range_1_box_list

            # Add heat to each box in box list
            heat = LF.add_heat(heat, box_list)

            # Apply threshold to help remove false positives
            heat = LF.apply_threshold(heat, 2)

            # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            draw_img = LF.draw_labeled_bboxes(np.copy(image), labels)

            # Display the results
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()
            
            
    def process_video_VD(image):

        spatial_size = (32,32)
        histbin = 32
        cell_per_block = 2
        pixels_per_cel = 8
        orient = 9
        color_space = 'HLS'
        with open('classifier.pkl', 'rb') as fid:
            svc = pickle.load(fid)
            
        with open('scaler.pkl', 'rb') as fid:
            X_scaler = pickle.load(fid)
            
            
        # Frames book-keeping
        if LF.frame_tracking >= 10:
            # Reset the number of frames counter
            LF.frame_tracking = 0
            
            # pass the full box list as valid for drawing
            LF.frame_seq_boxes = LF.single_frame_boxes[:]
            
            # empty box list
            LF.single_frame_boxes[:] = []
        
        # Increase the frame for the next itteration
        LF.frame_tracking = LF.frame_tracking + 1
        
        # Create an empty heat map to draw on
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        
        # Get the box list from using the hog sub sampling technique
        out_img, range_3_box_list = LF.find_cars(image,
                                             LF.y_start[3],
                                             LF.y_stop[3],
                                             LF.img_scale[3],
                                             svc, X_scaler,
                                             'ALL',
                                             orient,
                                             pixels_per_cel,
                                             cell_per_block,
                                             spatial_size,
                                             histbin,
                                            color_space=color_space)
            
        out_img, range_2_box_list = LF.find_cars(image,
                                             LF.y_start[2],
                                             LF.y_stop[2],
                                             LF.img_scale[2],
                                             svc, X_scaler,
                                             'ALL',
                                             orient,
                                             pixels_per_cel,
                                             cell_per_block,
                                             spatial_size,
                                             histbin,
                                            color_space=color_space)
            
        out_img, range_1_box_list = LF.find_cars(image,
                                              LF.y_start[1],
                                              LF.y_stop[1],
                                              LF.img_scale[1],
                                              svc, X_scaler,
                                              'ALL',
                                              orient,
                                              pixels_per_cel,
                                              cell_per_block,
                                              spatial_size,
                                              histbin,
                                             color_space=color_space)
        
        out_img, range_0_box_list = LF.find_cars(image,
                                                 LF.y_start[0],
                                                 LF.y_stop[0],
                                                 LF.img_scale[0],
                                                 svc, X_scaler,
                                                 'ALL',
                                                 orient,
                                                 pixels_per_cel,
                                                 cell_per_block,
                                                 spatial_size,
                                                 histbin,
                                                 color_space=color_space)
        
        # Append the local and global box list
        box_list = range_3_box_list + range_2_box_list + range_1_box_list + range_0_box_list
        LF.single_frame_boxes += box_list

        # Add heat to each box in box list
        heat = LF.add_heat(heat, LF.frame_seq_boxes)

        # Apply threshold to help remove false positives
        heat = LF.apply_threshold(heat, 12)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = LF.draw_labeled_bboxes(np.copy(image), labels)

        # Return the image with the detected vehicles
        return draw_img