import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Data fotografer
data_fotografer = np.array([
    [25, 2],    # Usia 25, Pengalaman 2 tahun
    [30, 5],    # Usia 30, Pengalaman 5 tahun
    [35, 7],    # Usia 35, Pengalaman 7 tahun
    [20, 1],    # Usia 20, Pengalaman 1 tahun
    [40, 10],   # Usia 40, Pengalaman 10 tahun
    [60, 15]    # Usia 60, Pengalaman 15 tahun
])

# Kualitas fotografer
kualitas_fotografer = np.array([4, 4, 3, 3, 5, 3])

# Membuat objek KNN regressor dengan k=3
knn = KNeighborsRegressor(n_neighbors=2)

# Melatih model menggunakan data kualitas fotografer
knn.fit(data_fotografer, kualitas_fotografer)

# Data fotografer baru untuk diprediksi
fotografer_baru = np.array([[24, 3]])  # Usia 24, Pengalaman 3 tahun

# Melakukan prediksi kualitas untuk fotografer baru
hasil_prediksi = knn.predict(fotografer_baru)

# Menampilkan hasil prediksi
print("Hasil prediksi:", hasil_prediksi)
