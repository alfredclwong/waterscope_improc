from __future__ import print_function
import os
import cv2
import numpy as np

def subtract_background(images):
    fgbg = cv2.BackgroundSubtractorMOG2(history=500,varThreshold=4,bShadowDetection=False)
    fgbg = cv2.BackgroundSubtractorMOG()
    for i,image in enumerate(images):
        if i < 10:
            fg_mask = fgbg.apply(image,learningRate=10)
        else:
            fg_mask = fgbg.apply(image,learningRate=0)
        #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        #cv2.imshow('image',image)
        #cv2.namedWindow('fg_mask',cv2.WINDOW_NORMAL)
        #cv2.imshow('fg_mask',fg_mask)
        #cv2.waitKey(0)

sats = []
def find_bacteria(images, bw_images, bacteria_hue, max_count=500, hue_range=20, min_radius=10):
    # for each possible bacteria (up to max_count), store its position and size in each image (-1 if not detected)
    bacteria_locations = -1*np.ones((max_count,len(images),3))
    
    # remember the number of bacteria counted in each image
    bacteria_counts = []
    
    for i,image in enumerate(images):
        # idea: narrow thresholding bounds over time
        #hue_range -= 0.1
        print("Processing image {0}/{1} - hue range {2} to {3}".format(i+1,len(images),
                                                                       bacteria_hue - hue_range / 2,
                                                                       bacteria_hue + hue_range / 2),end='\r')

        # denoise
        hsv_image = cv2.GaussianBlur(image, (5,5), 0)
        
        # convert bgr->hsv
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

        # idea: investigate regions of increasing saturation value
        sats.append(hsv_image[836,582,1])
        
        # threshold using hue
        # idea: use saturation value investigation to also threshold using saturation
        hue_mask = cv2.inRange(hsv_image,(bacteria_hue-hue_range/2,20,0), (bacteria_hue+hue_range/2,255,255))
        
        # 'erode' (more like a dilation, but the values are reversed) to eliminate small noise blobs
        hue_mask = cv2.erode(hue_mask, (1,1), iterations=5)

        bw_images.append(hue_mask)
        
        # find contours
        contours,_ = cv2.findContours(hue_mask.copy(),1,2)
        
        count = 0
        for contour in contours:
            # check for contours that contain more than 1 previously identified bacteria. if so, split contour.
            prev_bacteria_in_contour = 0
            if i > 0 and bacteria_counts[i-1] > 0:
                for bacteria in range(bacteria_counts[i-1]):
                    (x,y) = bacteria_locations[bacteria,i-1,:2]
                    radius = cv2.pointPolygonTest(contour,(x,y),True)
                    if radius >= 0:  #
                        bacteria_locations[count,i,:] = [x,y,max(radius,bacteria_locations[bacteria,i-1,2])]
                        prev_bacteria_in_contour += 1
                        count += 1
                        if count >= max_count:
                            break
                if count >= max_count:
                    break
            
            # otherwise define bacteria using min enclosing circle (if prev_bac_in_cnt is 1, don't double-count)
            if prev_bacteria_in_contour <= 1:
                (x,y),radius = cv2.minEnclosingCircle(contour)
                if radius >= min_radius:
                    bacteria_locations[count,i,:] = [x,y,radius]
                    if prev_bacteria_in_contour == 0:
                        count += 1
                        if count >= max_count:
                            break

        bacteria_counts.append(count)
        
        # draw green circles around every contour
        # draw blue convex hulls around every contour
        hulls = []
        for contour in contours:
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0,255,0), 3)

            hulls.append(cv2.convexHull(contour,False))
        cv2.drawContours(image,hulls,-1,(255,0,0),2)

        # draw red circles representing detected bacteria
        for bacteria in range(count):
            center = (int(bacteria_locations[bacteria,i,0]),
                      int(bacteria_locations[bacteria,i,1]))
            radius = int(bacteria_locations[bacteria,i,2])
            cv2.circle(image, center, radius, (0,0,255), 2)
        
        cv2.putText(image, str(count), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imwrite("count_{:3d}.png".format(i),image)
        cv2.imwrite("threshold_{:3d}.png".format(i),hue_mask)

        # cv2.namedWindow('cropped image',cv2.WINDOW_NORMAL)
        # cv2.imshow('cropped image',image)
        # cv2.namedWindow('hue mask',cv2.WINDOW_NORMAL)
        # cv2.imshow('hue mask',hue_mask)
        # cv2.waitKey(0)

    print('\n')


# Test 8: hue = 92
# Test 17: hue = 23
# Test 21: hue = 96

path = os.path.join(os.getcwd(), "Test 8")
images = []
bw_images = []
print("Reading images...")
for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        print(filename,end="\r")
        image = cv2.imread(os.path.join(path,filename))
        image = image[470:1790,1005:2325,:]
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (660,660), 660, (255,255,255), -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        images.append(image)
print("\nDone reading.")

subtract_background(images)

find_bacteria(images,bw_images,92)
print(sats)
current = 0
while True:
   cv2.namedWindow('',cv2.WINDOW_NORMAL)
   cv2.imshow('',images[current])
   cv2.namedWindow('hue thresholding',cv2.WINDOW_NORMAL)
   cv2.imshow('hue thresholding',bw_images[current])
   key = cv2.waitKey(0)
   if key == 27: # Esc
       break
   if key == 2424832: # left arrow
       current = max(0,current-1)
   elif key == 2555904: # right arrow
       current = min(len(images)-1,current+1)
