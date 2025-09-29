# 🚩 Penerjemah Bahasa Isyarat Semaphore 🚩

Aplikasi web inovatif yang mampu menerjemahkan gambar pose semaphore menjadi teks secara *real-time*. Unggah gambar Anda dan biarkan teknologi machine learning kami mengungkap pesan di baliknya!

![Demo Aplikasi Semaphore](https://img.shields.io/badge/status-aktif-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-Flask-orange)
![ML Library](https://img.shields.io/badge/library-TensorFlow%20%7C%20MediaPipe-red)

---

## ✨ Fitur Utama

-   **Unggah Gambar Interaktif**: Antarmuka yang ramah pengguna memungkinkan Anda mengunggah hingga 6 gambar pose semaphore dengan mudah.
-   **Deteksi Pose Akurat**: Didukung oleh **MediaPipe Pose**, aplikasi ini secara presisi mendeteksi 12 titik kunci (landmark) pada tubuh untuk analisis pose yang mendalam.
-   **Prediksi Cerdas**: Sebuah model Neural Network yang telah dilatih dengan TensorFlow/Keras secara cerdas memprediksi huruf dari setiap pose yang terdeteksi.
-   **Hasil Visual**: Setiap gambar yang diunggah ditampilkan kembali beserta huruf hasil terjemahannya, memberikan umpan balik yang jelas.
-   **Konstruksi Kalimat Otomatis**: Huruf-huruf yang berhasil diterjemahkan akan secara otomatis digabungkan menjadi sebuah kata atau kalimat yang utuh.
-   **Mulai Ulang dengan Mudah**: Fitur sekali klik untuk mereset aplikasi, menghapus semua gambar, dan memulai sesi terjemahan baru.

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
| :--- | :--- |
| **Backend** | Flask |
| **Machine Learning** | TensorFlow/Keras, MediaPipe, Scikit-learn, NumPy |
| **Frontend** | HTML, CSS, Bootstrap 5 |
| **Lainnya** | OpenCV |

## ⚙️ Bagaimana Cara Kerjanya?

Proses penerjemahan dilakukan dalam beberapa langkah kunci di dalam backend:

1.  **Deteksi Landmark Pose**:
    Saat gambar diunggah, `app.py` menggunakan `mediapipe.solutions.pose` untuk mendeteksi landmark tubuh. Kami fokus pada 12 koordinat dari 6 titik kunci untuk mendapatkan representasi pose yang akurat.

    ```python
    # Keypoints yang diekstrak dari pose
    key_points = {
        'left_hand': mp_pose.PoseLandmark.LEFT_WRIST,
        'right_hand': mp_pose.PoseLandmark.RIGHT_WRIST,
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'nose': mp_pose.PoseLandmark.NOSE,
    }
    ```

2.  **Normalisasi & Prediksi**:
    Koordinat (x, y) dari setiap landmark dinormalisasi dan diubah menjadi array NumPy. Array ini kemudian dimasukkan ke dalam model `landmark_model.h5`.

    ```python
    # Mempersiapkan data untuk prediksi
    X = np.array(coords).reshape(1, -1)
    prediction = model.predict(X)
    label_index = np.argmax(prediction)
    letter = label_encoder.inverse_transform([label_index])[0]
    ```

3.  **Tampilan Hasil**:
    Hasil prediksi (huruf) dikirim ke template `result.html` untuk ditampilkan kepada pengguna bersama dengan gambar aslinya.

## 🚀 Instalasi dan Menjalankan Proyek

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di lingkungan lokal Anda.

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/dly-k/smaphore-translate.git](https://github.com/dly-k/smaphore-translate.git)
    cd smaphore-translate
    ```

2.  **Buat Virtual Environment**
    Disarankan untuk menggunakan virtual environment untuk mengelola dependensi proyek.
    ```bash
    # Membuat venv
    python -m venv venv

    # Mengaktifkan venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi**
    Pasang semua library yang dibutuhkan. Anda bisa membuat file `requirements.txt` dengan isi berikut, lalu menjalankannya dengan `pip install -r requirements.txt`.
    
    **requirements.txt:**
    ```
    Flask
    opencv-python
    numpy
    mediapipe
    tensorflow
    scikit-learn
    ```

4.  **Jalankan Aplikasi**
    Mulai server pengembangan Flask.
    ```bash
    python app.py
    ```

5.  **Buka di Browser**
    Akses aplikasi melalui browser Anda di alamat `http://127.0.0.1:5000`.

## 🧠 Detail Model

Model yang digunakan adalah **Sequential Neural Network** yang dibangun dengan Keras. Arsitekturnya terdiri dari:
-   **Input Layer**: Menerima 12 fitur (koordinat x dan y dari 6 landmark).
-   **Hidden Layer 1**: Dense layer dengan 64 neuron dan fungsi aktivasi ReLU.
-   **Hidden Layer 2**: Dense layer dengan 64 neuron dan fungsi aktivasi ReLU.
-   **Output Layer**: Dense layer dengan 26 neuron (untuk 26 huruf alfabet) dan fungsi aktivasi Softmax untuk klasifikasi.

Model ini dilatih menggunakan data dari file `model/landmark_data.csv`.

---

Terima kasih telah mengunjungi proyek ini! Jangan ragu untuk memberikan bintang jika Anda menyukainya.
