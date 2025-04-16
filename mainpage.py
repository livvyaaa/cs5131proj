from flask import Flask, render_template
from landsat.app import app1
from kmeans.app import app2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
# Register the two Flask apps (as blueprints)
app.register_blueprint(app1, url_prefix='/app1')
app.register_blueprint(app2, url_prefix='/app2')

@app.route('/')
def home():
    return render_template('index.html')  # Intro tab or first page

if __name__ == '__main__':
    app.run(debug=True)
for rule in app.url_map.iter_rules():
    print(f"Endpoint: {rule.endpoint} Method: {rule.methods} URL: {rule}")
