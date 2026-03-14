import cv2

# 1. Menginisialisasi objek VideoCapture untuk webcam (indeks 0)
cap = cv2.VideoCapture(0)

# 2. Mendapatkan resolusi (lebar dan tinggi) dari webcam
#    cap.get() mengambil properti dari objek VideoCapture
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi webcam: {webcam_width}x{webcam_height}")

# Jika ingin mengubah ukuran, tentukan resolusi yang diinginkan.
# Namun, untuk menyesuaikan dengan webcam, kita gunakan resolusi aslinya.
new_width = webcam_width
new_height = webcam_height

# 3. Memuat gambar yang ingin diubah ukurannya
# Ganti 'nama_gambar.jpg' dengan nama file gambar Anda
image_to_resize = cv2.imread('D:\\Python\\Mediapipe\\project_gaje\\img\\I need MORE pictures of him(1).jpg')

# Memeriksa apakah gambar berhasil dimuat
if image_to_resize is None:
    print("Error: Gambar tidak dapat dimuat. Pastikan nama file dan path sudah benar.")
else:
    # 4. Mengubah ukuran gambar menggunakan cv2.resize()
    # Argumen pertama adalah gambar sumber
    # Argumen kedua adalah tuple (lebar, tinggi) yang baru
    resized_image = cv2.resize(image_to_resize, (new_width, new_height))

    # 5. Menampilkan gambar asli dan gambar yang sudah diubah ukurannya
    cv2.imshow('Gambar Asli', image_to_resize)
    cv2.imshow('Gambar Setelah Resize (Seukuran Webcam)', resized_image)

    # 6. Menunggu tombol keyboard ditekan dan menutup semua jendela
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 7. Melepaskan objek VideoCapture
cap.release()