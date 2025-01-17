# Segmentation 
This is the code for the 5th assignment: Segmentation, for the course Vision and Image Processing.

## **Set up**

To run the script, install the required libraries first:

```bash
pip install -r requirements.txt
```

Afterwards, run the script using:

```bash
python -m code_segmentation
```

The script will produce the original images and the applied segmentation algorithm applied to them.

---

## **Features**

The script implements various image segmentation techniques and preprocessing methods:

### 1. **Image Preprocessing**
- Converts PNG images to JPEG format to address ICC profile issues.
- Reads grayscale images for segmentation.

### 2. **Segmentation Algorithms**
#### **K-Means Clustering**
- Segments images based on intensity clusters.
- Supports adjustable `k` values for experimentation.
- Visualizes results for multiple `k` values side-by-side.

#### **Otsu's Thresholding**
- Implements single-threshold, two-threshold, and three-threshold versions.
- Displays histograms with threshold markers for visualization.

#### **Noise Cleaning and Denoising**
- Removes noise from segmented images using a neighborhood-based approach.
- Supports 4-pixel and 8-pixel neighborhood systems.
- Allows customization of threshold ratios and iteration counts.

#### **Chan-Vese Segmentation**
- Performs advanced segmentation using level sets.
- Displays energy evolution during segmentation.

---

## **Workflow**
1. **Image Conversion**: Convert all PNG files in the input directory to JPEG and save them in a new directory.
2. **Segmentation Methods**:
   - Apply K-Means clustering for a range of `k` values.
   - Use Otsu's thresholding techniques to find optimal intensity thresholds.
3. **Noise Removal**: Denoise segmented images with varying parameters.
4. **Advanced Segmentation**: Perform Chan-Vese segmentation and visualize energy evolution.

---

## **Usage**
1. **Input Directory**: Place all input images in the `images` folder.
2. **Run Script**: Execute the script as shown in the setup section.
3. **Output**: View original and segmented images along with visualizations in the output directory.

---

## **Dependencies**
Ensure the following Python libraries are installed:
- `numpy`
- `matplotlib`
- `opencv-python`
- `scikit-image`
- `glob`

These are included in the `requirements.txt` file.

---

## **Output Examples**
- Original images with segmentation overlays.
- Histograms displaying intensity thresholds.
- Energy evolution plots for Chan-Vese segmentation.

---