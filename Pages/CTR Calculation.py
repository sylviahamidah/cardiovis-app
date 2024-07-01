import streamlit as st
import cv2
import numpy as np


st.title(":three: Cardiothoracic Ratio (CTR) Calculation")


def ctr_line(heart, lung):
    heart = cv2.normalize(heart, None, 0, 255, cv2.NORM_MINMAX)
    heart = heart.astype(np.uint8)
    lung = cv2.normalize(lung, None, 0, 255, cv2.NORM_MINMAX)
    lung = lung.astype(np.uint8)
    
    ret, lung_thresh = cv2.threshold(lung, 20, 255, cv2.THRESH_BINARY)
    ret, heart_thresh = cv2.threshold(heart, 20, 255, cv2.THRESH_BINARY)
    
    # heart
    heart_nzp = np.argwhere(heart_thresh > 0)
    left_heartx = np.min(heart_nzp[:, 1])
    right_heartx = np.max(heart_nzp[:, 1])
    heart_distance = right_heartx - left_heartx

    # Find the corresponding y-coordinates
    y_heart = heart_nzp[:, 0]
    left_hearty = y_heart[np.where(heart_nzp[:, 1] == left_heartx)[0][0]]
    right_hearty = y_heart[np.where(heart_nzp[:, 1] == right_heartx)[0][0]]
    
    # Find the coordinates of non-zero pixels in the mask
    lung_nzp = np.argwhere(lung_thresh > 0)
    left_lungx = np.min(lung_nzp[:, 1])
    right_lungx = np.max(lung_nzp[:, 1])

    # Calculate the distance between the most left and right pixels
    lung_distance = right_lungx - left_lungx
    y_lung = lung_nzp[:, 0]
    left_lungy = y_lung[np.where(lung_nzp[:, 1] == left_lungx)[0][0]]
    right_lungy = y_lung[np.where(lung_nzp[:, 1] == right_lungx)[0][0]]

    heart_y = round((left_hearty+right_hearty)/2)
    #heart_y = np.max(y_heart)
    lung_y = np.max(y_lung)
    
    return left_heartx, heart_y, right_heartx, left_lungx, lung_y, right_lungx

def ctr_calc(heart, lung):
    heart = cv2.normalize(heart, None, 0, 220, cv2.NORM_MINMAX)
    heart = heart.astype(np.uint8)
    lung = cv2.normalize(lung, None, 0, 220, cv2.NORM_MINMAX)
    lung = lung.astype(np.uint8)
    
    ret, lung_thresh = cv2.threshold(lung, 20, 255, cv2.THRESH_BINARY)
    ret, heart_thresh = cv2.threshold(heart, 5, 255, cv2.THRESH_BINARY)
    
    # heart
    heart_nzp = np.argwhere(heart_thresh > 0)
    left_heartx = np.min(heart_nzp[:, 1])
    right_heartx = np.max(heart_nzp[:, 1])
    heart_distance = right_heartx - left_heartx

    # Find the corresponding y-coordinates
    y_heart = heart_nzp[:, 0]
    
    # Find the coordinates of non-zero pixels in the mask
    lung_nzp = np.argwhere(lung_thresh > 0)
    left_lungx = np.min(lung_nzp[:, 1])
    right_lungx = np.max(lung_nzp[:, 1])

    # Calculate the distance between the most left and right pixels
    lung_distance = right_lungx - left_lungx
    y_lung = lung_nzp[:, 0]
    ctr = heart_distance/lung_distance
    ctr_rounded = round(ctr, 5)
    
    return heart_distance, lung_distance, ctr_rounded

# heart mask
def heart_mask(image_base, heart):
    heart = cv2.normalize(heart, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to the range 0-255
    heart = heart.astype(np.uint8)
    lowerb = np.array(200, dtype=np.uint8)
    upperb = np.array(255, dtype=np.uint8)
    
    mask = cv2.inRange(heart, lowerb, upperb)
    heart_rgb = cv2.cvtColor(heart, cv2.COLOR_GRAY2BGR)
    heart_rgb[mask == 255] = [255, 0, 0]
    alpha = 1  # Opacity untuk img1
    beta = 0.7   # Opacity untuk img2
    gamma = 0 
    result1 = cv2.addWeighted(image_base, alpha, heart_rgb, beta, gamma)
    return result1
    
# lung mask
def lung_mask(image_base, lung):
    lung = cv2.normalize(lung, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to the range 0-255
    lung = lung.astype(np.uint8)
    lowerb = np.array(200, dtype=np.uint8)
    upperb = np.array(255, dtype=np.uint8)
    
    lung_mask = cv2.inRange(lung, lowerb, upperb)
    lung_rgb = cv2.cvtColor(lung, cv2.COLOR_GRAY2BGR)
    lung_rgb[lung_mask == 255] = [128, 0, 128]
    alpha = 1  # Opacity untuk img1
    beta = 0.5   # Opacity untuk img2
    gamma = 0    # Skalar opsional tambahan
    result2 = cv2.addWeighted(image_base, alpha, lung_rgb, beta, gamma)
    return result2
    
def draw_line(image_base, heart, lung):
    heart = cv2.normalize(heart, None, 0, 100, cv2.NORM_MINMAX)
    heart = heart.astype(np.uint8)
    lung = cv2.normalize(lung, None, 0, 100, cv2.NORM_MINMAX)
    lung = lung.astype(np.uint8)
    left_heartx, heart_y, right_heartx, left_lungx, lung_y, right_lungx = ctr_line(heart, lung)
    # st.write(left_heartx)
    # st.write(right_heartx)
    # st.write(left_lungx)
    # st.write(right_lungx)
    
    lung_line = np.zeros_like(lung)
    heart_line = np.zeros_like(heart)
    color = (180, 0, 180)
    thickness = 2
    cv2.line(heart_line, (left_heartx, heart_y), (right_heartx, heart_y), color, thickness)
    cv2.line(lung_line, (left_lungx, lung_y), (right_lungx, lung_y), color, thickness)

    heart_line = cv2.cvtColor(heart_line, cv2.COLOR_GRAY2BGR)
    lung_line = cv2.cvtColor(lung_line, cv2.COLOR_GRAY2BGR)
    
    # Atur tingkat transparansi
    alpha = 1  # Opacity untuk img1
    beta = 1   # Opacity untuk img2
    gamma = 0    # Skalar opsional tambahan
    result3 = cv2.addWeighted(image_base, alpha, heart_line, beta, gamma)
    result4 = cv2.addWeighted(result3, alpha, lung_line, beta, gamma)
    return result4

col1, col2 = st.columns([0.3, 0.7], gap='medium')
with col1:
    st.markdown('#### Display Option')
    st.write("Choose what to display")
    segmentation = st.checkbox("Image Segmentation")
    line = st.checkbox("Diameter Line")
    ctr_calculation = st.checkbox("CTR Calculation")
    apply = st.button("Apply", type="secondary")
    st.write("OR")
    run_all = st.button("Run All", type="secondary")
    
    if run_all:
        segmentation = True
        line = True
        ctr_calculation = True
        apply = True

with col2:
    st.markdown('#### Output Images')
    if 'img_resize' in st.session_state:
        img_resize = st.session_state['img_resize']
    if 'heart_output' in st.session_state:
        heart_output = st.session_state['heart_output']
    if 'left_output' in st.session_state:
        left_output = st.session_state['left_output']
    if 'right_output' in st.session_state:
        right_output = st.session_state['right_output']
        
    img_array = np.array(img_resize)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # lung
    alpha = 1  # Opacity untuk img1
    beta = 1   # Opacity untuk img2
    gamma = 0    # Skalar opsional tambahan
    lung_output = cv2.addWeighted(left_output, alpha, right_output, beta, gamma)
            
    if apply:
        if segmentation and line:
            image_base = img_rgb
            result1 = heart_mask(image_base, heart_output)
            result2 = lung_mask(result1, lung_output) 
            result3 = draw_line(result2, heart_output, lung_output)
            st.image(result3, width=300)
            
        elif segmentation:
            image_base = img_rgb
            result1 = heart_mask(image_base, heart_output)
            result2 = lung_mask(result1, lung_output)
            st.image(result2, width=300)
        
        elif line:
            image_base = img_rgb
            result1 = draw_line(image_base, heart_output, lung_output)
            st.image(result1, width=300)
        
        if ctr_calculation:
            mhcd, mhtd, ctr = ctr_calc(heart_output, lung_output)
            st.markdown('#### Result')
            st.write("Maximum Horizontal Cardiac Diameter (MHCD): ", mhcd)
            st.write("Maximum Horizontal Thorax Diameter (MHTD): ", mhtd)
            st.write("Cardiothoracic Ratio (CTR): ", ctr)
            
            if ctr > 0.5:
                st.write("The patient has been diagnosed with cardiomegaly.")
            else:
                st.write("The patient has not been diagnosed with cardiomegaly.")
            
        
