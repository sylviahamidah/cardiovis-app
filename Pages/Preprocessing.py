import streamlit as st
from PIL import Image
import opencv-python as cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from contrast_image import quantitation
quantitation = quantitation.Quantitation()

st.title(":one: Image Preprocessing")
# ------------------------------------------------------------------------------
# upload file
st.markdown('#### Upload File')
file_uploaded = st.file_uploader("Select a file from your computer", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False,
                 help="Drag thorax x-ray image here", on_change=None, label_visibility="visible")

ex_image = st.radio(
    "OR select from the example images below:",
    ["Normal", "Cardiomegaly", "Cardiomegaly + pleural effusion"],
    horizontal=True, index=None)

if file_uploaded is not None:
    #st.write("filename:", file_uploaded.name)
    image = Image.open(file_uploaded)  # Convert the UploadedFile to a PIL Image
    #st.image(image, caption='Raw Image', width=200)
elif ex_image == "Normal":
    image = Image.open("Findingno_fix.png")
elif ex_image == "Cardiomegaly":
    image = Image.open("cardiom2.png")
elif ex_image == "Cardiomegaly + pleural effusion":
    image = Image.open("cardiop4.png")


st.markdown('######')
# ------------------------------------------------------------------------------
# function for intensity thresholding
def calculate_window(image):
    # Calculate window center and window width based on percentiles
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    # window_center = (min_intensity + max_intensity) / 2 
    window_center = (min_intensity / 2) + (max_intensity / 2)
    window_width = max_intensity - min_intensity
    return window_center, window_width

def apply_custom_threshold(image_array, thresh_min, thresh_max):
    # Initialize an output array with zeros (black)
    result_array = np.zeros_like(image_array, dtype=np.float32)
    # Apply the custom thresholding rules
    result_array[image_array < thresh_min] = 0
    result_array[image_array > thresh_max] = 255
    in_range = (image_array >= thresh_min) & (image_array <= thresh_max)
    result_array[in_range] = ((image_array[in_range] - thresh_min) * 255) / (thresh_max - thresh_min)
    return result_array.astype(np.uint8)

# ------------------------------------------------------------------------------
col1, col2 = st.columns([0.3, 0.7], gap='medium')

with col1:
    st.markdown('#### Preprocessing Method')
    st.write("Choose your preferred method:")
    resize = st.checkbox("Resize")
    clahe = st.checkbox("Contrast Limited Adaptive Histogram Equalization")
    thres = st.checkbox("Intensity Thresholding")
    canny = st.checkbox("Canny Edge Detection")
    apply = st.button("Apply", type="secondary")
    #next_step = st.button("Use Output Images to Next Step", type="secondary")
    
    st.write("OR")
    run_all = st.button("Run All", type="secondary")
    
    if run_all:
        resize = True
        clahe = True
        thres = True
        canny = True
        apply = True
        
    st.session_state['next_step'] = True
        #st.experimental_rerun()
    
    
with col2:
    st.markdown('#### Output Images')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Raw", "Resize", "CLAHE", "Thresholding", "Canny"])
    
    with tab1:
        st.image(image, width=350)
        image_array = np.array(image)
        shape_image = image_array.shape
        st.write("The shape of the raw image is: ", shape_image)
        
    if apply:
        if resize:
            img_resize = image
            img_resize = img_resize.resize((256, 256))
            st.session_state['img_resize'] = img_resize
            img_resize = img_resize.convert('L')
            with tab2:
                st.image(img_resize, width=350)
                imgresize_array = np.array(img_resize)
                shape_resize = imgresize_array.shape
                st.write("The shape of the resized image is: ", shape_resize)
        
        if clahe:
            if resize:
                img_clahe = img_resize
            else:
                img_clahe = image
            img_clahe = np.array(img_clahe)       
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img_clahe)
            psnr = cv.PSNR(imgresize_array, img_clahe)
            ssim = compare_ssim(imgresize_array, img_clahe)
            ambe = quantitation.AMBE(imgresize_array, img_clahe)
            global_variance = np.var(img_clahe)
            with tab3:
                st.image(img_clahe, width=350)
                st.write("Peak Signal to Noise Ratio (PSNR): ", psnr)
                st.write("Structural Similarity Index Measure (SSIM): ", ssim)
                st.write("Absolute Mean Brightness Error (AMBE): ", ambe)
                st.write("Global Variance: ", global_variance)
        
        if thres:
            if resize and clahe:
                img_thres = img_clahe
            elif resize:
                img_thres = img_resize
            elif clahe:
                img_thres = img_clahe
            else:
                img_thres = image
            img_thres = np.array(img_thres)
            window_center, window_width = calculate_window(img_thres)
            min_threshold = window_center - 0.5 * window_width
            max_threshold = window_center + 0.5 * window_width
            img_thres = apply_custom_threshold(img_thres, min_threshold, max_threshold)
            psnr_th = cv.PSNR(img_clahe, img_thres)
            ssim_th = compare_ssim(img_clahe, img_thres)
            ambe_th = quantitation.AMBE(img_clahe, img_thres)
            global_variance_th = np.var(img_thres)
            with tab4:
                st.image(img_thres, width=350)
                # st.write("Peak Signal to Noise Ratio (PSNR): ", psnr_th)
                # st.write("Structural Similarity Index Measure (SSIM): ", ssim_th)
                # st.write("Absolute Mean Brightness Error (AMBE): ", ambe_th)
                # st.write("Global Variance: ", global_variance_th)
        
        if canny:
            if resize and clahe and thres:
                img_canny = img_thres
            elif resize and clahe:
                img_canny = img_clahe
            elif resize:
                img_canny = img_resize
            elif clahe:
                img_canny = img_clahe
            elif thres:
                img_canny = img_thres
            else:
                img_canny = image
            img_canny = np.array(img_canny)
            #img_gray = cv.cvtColor(img_canny, cv.COLOR_BGR2GRAY)
            ret, thresh1 = cv.threshold(img_canny, 0, 255, cv.THRESH_OTSU)
            median = cv.medianBlur(thresh1, 3)
            canny_edges = cv.Canny(median, threshold1=0.6*ret, threshold2=ret)
            #img_bgr = cv.cvtColor(img_canny, cv.COLOR_BGR2GRAY)
            result = cv.bitwise_or(img_canny, canny_edges)
            st.session_state['result'] = result
            with tab5:
                st.image(result, width=350)
        
    
    
        
