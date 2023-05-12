import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Automatic image histogram EQ
def image_eq_basic():
    # Upload an image
    source_image = st.file_uploader('Upload an image', type=['JPG', 'JPEG', 'PNG'])

    # Placeholders for future use
    imageholder_1, imageholder_2 = st.columns([1, 1])
    histholder_1, histholder_2 = st.columns([1, 1])

    # When image is uploaded
    if source_image is not None:

        # Convert an image to cv2-acceptable format
        image_raw_bytes = np.asarray(bytearray(source_image.read()), dtype=np.uint8)
        source_image = cv2.imdecode(image_raw_bytes, cv2.IMREAD_COLOR)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)

        imageholder_1.image(source_image)
        imageholder_1.text('Input image')

        if choice_color == 'Grayscale':
            output_image = cv2.imdecode(image_raw_bytes, cv2.IMREAD_GRAYSCALE)
            output_image = cv2.equalizeHist(output_image)
        elif choice_color == 'Color':
            output_image = cv2.imdecode(image_raw_bytes, cv2.IMREAD_COLOR)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
            output_image[:, :, 2] = cv2.equalizeHist(output_image[:, :, 2])
            output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2RGB)

        imageholder_2.image(output_image)
        imageholder_2.text('Equalized image')

        # Display image histograms
        choice_hist = st.checkbox('Display image histograms')

        if choice_hist:
            hist_1 = plt.figure(figsize=(5, 5))
            plt.hist(source_image.flatten(), bins=255, range=[0, 256])
            plt.xlabel('Pixel intensity')
            plt.ylabel('Number of pixels')

            hist_2 = plt.figure(figsize=(5, 5))
            plt.hist(output_image.flatten(), bins=255, range=[0, 256])
            plt.xlabel('Pixel intensity')
            plt.ylabel('Number of pixels')

            histholder_1.pyplot(fig=hist_1)
            histholder_2.pyplot(fig=hist_2)

        st.text(f"Total: {source_image.shape[0] * source_image.shape[1]} pixels.")


# Manual CLAHE EQ
def image_eq_clahe():
    # Upload an image
    source_image = st.file_uploader('Upload an image', type=['JPG', 'JPEG', 'PNG'])

    # Placeholders for future use
    imageholder_1, imageholder_2 = st.columns([1, 1])
    sliderholders = st.columns(2)

    # When image is uploaded
    if source_image is not None:

        # Convert an image to cv2-acceptable format
        image_raw_bytes = np.asarray(bytearray(source_image.read()), dtype=np.uint8)
        source_image = cv2.imdecode(image_raw_bytes, cv2.IMREAD_COLOR)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)

        imageholder_1.image(source_image)
        imageholder_1.text('Input image')

        clip_limit = sliderholders[0].slider('Amount of light:', min_value=1, max_value=50, step=1)
        tile_grid = sliderholders[1].slider('Distribution of light:', min_value=1, max_value=50, step=1)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

        if choice_color == 'Grayscale':
            output_image = cv2.imdecode(image_raw_bytes, cv2.IMREAD_GRAYSCALE)
            output_image = cv2.equalizeHist(output_image)

        elif choice_color == 'Color':
            output_image = cv2.imdecode(image_raw_bytes, cv2.IMREAD_COLOR)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
            output_image[:, :, 2] = clahe.apply(output_image[:, :, 2])
            output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2RGB)

        imageholder_2.image(output_image)
        imageholder_2.text('Equalized image')

        # Display image histograms
        choice_hist = st.checkbox('Display image histograms')

        if choice_hist:
            histholder_1, histholder_2 = st.columns([1, 1])

            hist_1 = plt.figure(figsize=(5, 5))
            plt.hist(source_image.flatten(), bins=255, range=[0, 256])
            plt.xlabel('Pixel intensity')
            plt.ylabel('Number of pixels')

            hist_2 = plt.figure(figsize=(5, 5))
            plt.hist(output_image.flatten(), bins=255, range=[0, 256])
            plt.xlabel('Pixel intensity')
            plt.ylabel('Number of pixels')

            histholder_1.pyplot(fig=hist_1)
            histholder_2.pyplot(fig=hist_2)

        st.info(f"Total: {source_image.shape[0] * source_image.shape[1]} pixels.")


# Streamlit GUI
st.set_page_config(layout="wide")

st.title('Computer Vision Image Histogram Equalization')

choice_1 = st.radio('What would you like to do?', ['Nothing', 'Image Histogram Equalization'])

if choice_1 == 'Image Histogram Equalization':

    choice_eq = st.selectbox('Specify equalization type:',
                             ['Automatic Equalization', 'Contrast Limited Adaptive Histogram Equalization'],
                             help="Automatic Equalization distributes color hue based on pixel intensity. " \
                                  "By using Contrast Limited Adaptive Histogram Equalization (CLAHE), " \
                                  "you can manually adjust the amount of light and its distribution " \
                                  "to get the best result.")

    choice_color = st.selectbox('Desired output:', ['Color', 'Grayscale'],
                                help="Select 'Color' to keep the input image in color. "
                                     "Select 'Grayscale' to convert it to black & white.")

    if choice_eq == 'Automatic Equalization':
        image_eq_basic()

    elif choice_eq == 'Contrast Limited Adaptive Histogram Equalization':
        image_eq_clahe()
