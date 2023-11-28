from flask import Flask, render_template, request
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Membaca data dari file Excel
file_path = 'hasil_klasterisasi_Palang_All.xlsx'  # Gantilah dengan path sesuai lokasi file Anda
df = pd.read_excel(file_path)

# Memilih fitur
features = df.iloc[:, 2:-2]  # Memilih kolom ke-3 hingga sebelum kolom terakhir

# Standarisasi fitur karena Gaussian Naive Bayes memerlukan asumsi bahwa data terdistribusi normal
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Membuat model Gaussian Naive Bayes
target = df['Klaster']  # Menggunakan kolom 'Klaster' sebagai target
model = GaussianNB()
model.fit(features_scaled, target)

# Fungsi untuk melakukan prediksi
def predict(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return prediction[0]

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk hasil prediksi
@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # Mengambil input dari formulir
        input_features = [float(request.form['feature1']), float(request.form['feature2']), float(request.form['feature3']),
                          float(request.form['feature4']), float(request.form['feature5']), float(request.form['feature6'])]

        # Melakukan prediksi
        result = predict(input_features)

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
