# 🖐️ Belajar MediaPipe by Hanif Fawwaz

Belajar **Computer Vision menggunakan MediaPipe + Python** dari **dasar sampai project nyata**.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-00A651?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-white?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

---

# 📌 Overview

Repository ini berisi kumpulan **tutorial, eksperimen, dan project** untuk mempelajari **MediaPipe menggunakan Python**.

MediaPipe adalah framework dari Google yang memudahkan developer membangun aplikasi **computer vision dan AI perception pipeline** seperti:

* Face Detection
* Hand Tracking
* Pose Estimation
* Object Tracking
* Gesture Recognition

Framework ini dirancang untuk membantu developer membuat prototype hingga aplikasi production secara lebih mudah. ()

Repository ini dibuat sebagai **catatan belajar sekaligus panduan praktis** untuk memahami MediaPipe dari dasar.

---

# 🎯 Tujuan Repository

Tujuan repo ini adalah untuk:

* 📚 Belajar MediaPipe dari **dasar**
* 🧠 Memahami konsep **computer vision**
* ⚙️ Membuat **project AI real-time**
* 🤖 Mengembangkan sistem seperti:

  * Face Recognition
  * Gesture Control
  * AI Camera
  * Human Pose Detection

---

# ✨ Materi yang Dipelajari

Berikut beberapa topik yang dibahas dalam repo ini.

### 👤 Face Detection

Mendeteksi wajah secara real-time menggunakan webcam.

Contoh penggunaan:

* Face bounding box
* Confidence score
* Multiple face detection

---

### ✋ Hand Tracking

MediaPipe dapat melacak **21 landmark tangan** secara real-time.

Fitur:

* Deteksi jari
* Gesture recognition
* Finger counting

---

### 🧍 Pose Detection

Melacak posisi tubuh manusia menggunakan **body landmarks**.

Digunakan untuk:

* Fitness tracker
* Motion analysis
* Gesture control

---

### 👁 Face Mesh

Face mesh dapat mendeteksi **468 titik landmark wajah**.

Digunakan untuk:

* AR filter
* Face tracking
* Emotion analysis

---

# 🧠 How MediaPipe Works

```
Camera / Video Input
        │
        ▼
Frame Processing
        │
        ▼
MediaPipe ML Model
        │
        ▼
Landmark Detection
        │
        ▼
Visualization (OpenCV)
```

MediaPipe memproses frame video secara real-time untuk mendeteksi objek atau bagian tubuh tertentu menggunakan model machine learning. ()

---

# 🛠 Tech Stack

| Technology | Function                |
| ---------- | ----------------------- |
| Python     | Programming language    |
| OpenCV     | Video processing        |
| MediaPipe  | AI perception framework |
| NumPy      | Numerical computation   |

---

# 📦 Requirements

Pastikan sudah menginstall:

* Python **3.8+**
* pip
* Webcam

---

# 🚀 Installation

## 1️⃣ Clone Repository

```
git clone https://github.com/hanif-fawwaz/Belajar-mediapipe-by-hanif-fawwaz.git
cd Belajar-mediapipe-by-hanif-fawwaz
```

---

## 2️⃣ Install Dependencies

```
pip install mediapipe opencv-python numpy
```

atau menggunakan requirements file:

```
pip install -r requirements.txt
```

---

# ▶️ Menjalankan Project

Contoh menjalankan script:

```
python face_detection.py
```

atau

```
python hand_tracking.py
```

Pastikan webcam sudah aktif.

---

# 📂 Struktur Repository

```
Belajar-mediapipe-by-hanif-fawwaz
│
├── face_detection.py
│
├── hand_tracking.py
│
├── pose_detection.py
│
├── face_mesh.py
│
├── requirements.txt
│
└── README.md
```

---

# 📸 Contoh Output

Beberapa hasil yang bisa dibuat dari project ini:

* 👤 Real-time face detection
* ✋ Hand gesture tracking
* 🧍 Pose estimation
* 👁 Face landmark detection

---

# 🔮 Future Project

Beberapa project yang akan ditambahkan:

* [ ] Hand Gesture Controller
* [ ] AI Virtual Mouse
* [ ] Gesture Volume Control
* [ ] AI Fitness Trainer
* [ ] Face Recognition System
* [ ] Pose-based Game Control

---

# 🤝 Contributing

Kontribusi sangat terbuka!

Langkah kontribusi:

```
1. Fork repository
2. Buat branch baru
3. Commit perubahan
4. Push ke repository
5. Buat Pull Request
```

---

# 📜 License

Project ini menggunakan **MIT License**.

---

# 👨‍💻 Author

**Hanif Fawwaz**

GitHub
https://github.com/hanif-fawwaz

---

⭐ Jika repository ini membantu kamu belajar **Computer Vision**, jangan lupa beri **Star di GitHub!**
