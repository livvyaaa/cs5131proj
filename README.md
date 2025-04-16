# 🔥 Fire Analysis Tools

This repository contains two major components:

1. **Fire Point Data Tool** — A Google Earth Engine + Flask webapp for generating, uploading, visualizing, and clustering VIIRS-based fire points.
2. **Landsat Image Detection** — A deep learning model for classifying Landsat image patches (fire vs non-fire) using a trained CNN.

---

## Key Files for Grading:
-  [Random Forest + Exploratory Training](./testing/random.ipynb)
-  [CNN Model Training on Landsat](./landsat/cnn.ipynb)
-  [LSTM Time Series on ERA5 Weather Data](./era5_lstm_cleaned.ipynb)
## 🧩 PART 1: Fire Point Tool (Google Earth Engine + VIIRS)

### ✅ Features
- Fetch VIIRS fire data by country and date
- Visualize data as KDE heatmaps or scatterplots
- Cluster using K-Means and export centroid CSV
- Upload your own fire CSVs

### Main Landing Page
Go to `http://127.0.0.1:5000/`

### ▶️ Run Instructions
```bash
python mainpage.py
```
Then go to: `http://127.0.0.1:5000/app1`

### 🧠 CSV Format
```csv
lat,lon,date
-35.2,149.1,2020-01-01
-34.9,149.3,2020-01-01
```

---

## 🛰 PART 2: Landsat Image Detection (CNN Model)

This tool loads `.tif` Landsat image patches and uses a pre-trained CNN to predict whether a fire is present.

### 🏁 To Run the Webapp:
```bash
python mainpage.py
```
Then open: `http://127.0.0.1:5000/app2`

### 🖼 How It Works
- You upload `.tif` images (typically clipped Landsat 8 or Sentinel-2 RGB or SWIR bands)
- The app feeds the patch into `fire_detection_model.keras`
- Returns prediction: fire 🔥 or no fire 


### 🧠 Notes
- Model expects normalized 3-band input (shape: 256x256x3)
- Only `.tif` images are supported for upload

---

## 📦 Requirements
```bash
pip install flask pandas numpy seaborn matplotlib scikit-learn tensorflow rasterio
```
You may also need:
```bash
pip install earthengine-api geemap
```
And authenticate Earth Engine:
```bash
earthengine authenticate
```
