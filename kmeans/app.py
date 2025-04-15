import threading
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
import ee
import geemap
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io
import seaborn as sns
import numpy as np

# Initialize GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

app2 = Blueprint('app2', __name__, static_folder='static', template_folder='templates')

fetch_status = {}

@app2.route('/')
def app2_home():
    return render_template('dashboard.html')

@app2.route('/fetch_fire_data', methods=['POST'])
def fetch_fire_data():
    data = request.json
    country = data.get('country')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    fetch_id = f"{country}_{start_date}_{end_date}"

    fetch_status[fetch_id] = 'processing'

    def fetch_and_save():
        try:
            countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
            country_feat = countries.filter(ee.Filter.eq("ADM0_NAME", country)).first()

            viirs_fires = ee.ImageCollection("NOAA/VIIRS/001/VNP14A1") \
                .filterDate(start_date, end_date) \
                .filterBounds(country_feat.geometry()) \
                .select("MaxFRP")

            def image_to_vectors(image):
                frp = image.select("MaxFRP")
                mask = frp.gt(0).selfMask()

                vectors = mask.reduceToVectors(
                    geometry=country_feat.geometry(),
                    scale=1000,
                    geometryType='centroid',
                    labelProperty='burn',
                    reducer=ee.Reducer.countEvery(),
                    maxPixels=1e13
                )

                return vectors.map(lambda f: f.set('date', image.date().format('YYYY-MM-dd')))

            fire_points = viirs_fires.map(image_to_vectors).flatten()

            size = fire_points.size().getInfo()
            chunk_size = 3000
            num_chunks = (size // chunk_size) + 1

            all_features = []
            print(f"[INFO] Fetching {size} fire features in {num_chunks} chunks...")

            for i in range(num_chunks):
                start = i * chunk_size
                sublist = fire_points.toList(count=chunk_size, offset=start)
                fc_chunk = ee.FeatureCollection(sublist)

                try:
                    geojson = geemap.ee_to_geojson(fc_chunk)
                    features = [
                        {
                            'lon': f['geometry']['coordinates'][0],
                            'lat': f['geometry']['coordinates'][1],
                            'date': f['properties'].get('date', None)
                        } for f in geojson['features']
                    ]
                    all_features.extend(features)
                    print(f"[Chunk {i+1}/{num_chunks}] Retrieved {len(features)} features.")
                except Exception as e:
                    print(f"Error fetching chunk {i + 1}: {e}")

            df = pd.DataFrame(all_features)

            if df.empty:
                print("[WARNING] No fire data was retrieved. Returning empty CSV.")
                df = pd.DataFrame(columns=['lat', 'lon', 'date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            os.makedirs("static", exist_ok=True)
            filename = f"static/{fetch_id}_firepoints.csv"
            df.to_csv(filename, index=False)
            os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'uploads'), exist_ok=True)
            filename = os.path.join(os.path.dirname(__file__), 'static', 'uploads', f"{fetch_id}_firepoints.csv")
            df.to_csv(filename, index=False)
            fetch_status[fetch_id] = filename
            print(f"[INFO] CSV saved to {filename} with {len(df)} rows.")
        except Exception as e:
            fetch_status[fetch_id] = 'error'
            print(f"[ERROR] Fetching failed: {e}")

    threading.Thread(target=fetch_and_save).start()

    return jsonify({'status': 'started', 'fetch_id': fetch_id})

@app2.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        
        return jsonify({'status': 'success', 'file': filepath}), 200
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

@app2.route('/check_status/<fetch_id>', methods=['GET'])
def check_status(fetch_id):
    status = fetch_status.get(fetch_id, 'not_found')
    return jsonify({'status': status})

@app2.route('/heatmap')
def heatmap():
    # Get the CSV file path from the URL query parameter
    csv_path = request.args.get('csv')

    # Log the file path for debugging
    print(f"Requested CSV path: {csv_path}")

    # Check if the path is provided and if the file exists
    if not csv_path:
        return "CSV file path not provided", 400

    
    csv_full_path = os.path.join('static', csv_path)

    # Log to check if the file path is correct
    print(f"Full file path: {csv_full_path}")

    if not os.path.exists(csv_full_path):
        print(f"CSV file not found at: {csv_full_path}")  # Log if the file is not found
        return "CSV file not found", 404

    # Read the CSV file and remove rows with missing lat/lon values
    df = pd.read_csv(csv_full_path).dropna(subset=['lat', 'lon'])

    # Check if there is enough data to generate a heatmap
    if df.empty or len(df) < 10:
        return "Not enough data to generate heatmap", 400

    # Optionally, sample the data if there are too many points
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    # Create the heatmap using KDE (Kernel Density Estimation)
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=df['lon'], y=df['lat'], cmap="hot", fill=True, thresh=0.05)
    plt.title("Fire Location Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # Save the plot into a buffer and return it as a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app2.route('/scatter')
def scatter():
    csv_path = request.args.get('csv')
    if not csv_path or not os.path.exists(csv_path):
        return "CSV file not found", 404

    df = pd.read_csv(csv_path).dropna(subset=['lat', 'lon'])
    if df.empty:
        return "No valid data found", 400

    plt.figure(figsize=(10, 8))
    plt.scatter(df['lon'], df['lat'], s=10, alpha=0.6, color="#AB47BC") 
    plt.title("Fire Point Scatterplot")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app2.route('/scatter_with_centroids', methods=['POST'])
def scatter_with_centroids():
    data = request.json
    csv_path = data.get('csv')
    n_clusters = int(data.get('n_clusters'))

    if not csv_path or not os.path.exists(csv_path):
        return "CSV file not found", 404

    df = pd.read_csv(csv_path).dropna(subset=['lat', 'lon'])
    coords = df[['lon', 'lat']].values

    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(coords)
    centroids = km.cluster_centers_

    centroid_df = pd.DataFrame(centroids, columns=['lon', 'lat'])
    centroid_filename = f"static/centroids_{os.path.basename(csv_path).split('.')[0]}_{n_clusters}.csv"
    centroid_df.to_csv(centroid_filename, index=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(df['lon'], df['lat'], s=10, alpha=0.5, color="#CE93D8", label='Fire Points') 
    plt.scatter(centroids[:, 0], centroids[:, 1], c='#FF5252', edgecolors='black', s=120, marker='X', label='Centroids') 
    plt.title(f"Fire Points with {n_clusters} Cluster Centroids")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app2.route('/download_centroids', methods=['POST'])
def download_centroids():
    data = request.json
    csv_path = data.get('csv')
    n_clusters = int(data.get('n_clusters'))

    if not csv_path or not os.path.exists(csv_path):
        return "CSV file not found", 404

    df = pd.read_csv(csv_path).dropna(subset=['lat', 'lon'])
    coords = df[['lon', 'lat']].values
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(coords)
    centroids = km.cluster_centers_

    centroid_df = pd.DataFrame(centroids, columns=['lon', 'lat'])
    buf = io.StringIO()
    centroid_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.read().encode()), mimetype='text/csv', as_attachment=True, download_name=f"centroids_{n_clusters}.csv")

if __name__ == '__main__':
    app2.run(debug=True)
