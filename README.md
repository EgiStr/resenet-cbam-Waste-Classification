# ♻️ CBAM-ResNet Waste Classification

> **Sistem Klasifikasi Sampah Cerdas Berbasis Deep Learning dengan Integrasi Mekanisme Atensi**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Ikhtisar Proyek

Proyek ini mengimplementasikan solusi *Deep Learning* mutakhir untuk klasifikasi sampah otomatis, membedakan antara sampah **Organik** dan **Daur Ulang**. Inti dari solusi ini adalah arsitektur **ResNet-34** yang dimodifikasi dengan **Convolutional Block Attention Module (CBAM)**.

Integrasi CBAM memungkinkan model untuk memfokuskan "perhatian" pada fitur visual yang relevan (seperti bentuk dan tekstur objek) sambil menekan *noise* dari latar belakang yang kompleks. Hasilnya adalah model yang lebih akurat dan robust dibandingkan arsitektur CNN standar.

### ✨ Fitur Utama
*   **Arsitektur Hybrid**: ResNet-34 backbone dengan modul atensi CBAM pada setiap blok residual.
*   **Akurasi Tinggi**: Mencapai akurasi validasi **~92%** dan F1-Score **0.92**.
*   **Aplikasi Interaktif**: Antarmuka pengguna berbasis web yang intuitif menggunakan **Streamlit**.
*   **Reproducibility**: Pipeline pelatihan lengkap dengan logging, checkpointing, dan konfigurasi benih acak.

---

## 🏗️ Struktur Proyek

```
resenet-cbam/
├── config/                 # File konfigurasi
├── data/                   # Direktori dataset (Train/Val/Test)
├── docs/                   # Dokumentasi teknis & draft paper
│   ├── architecture_details.md
│   └── paper_draft.md
├── models/                 # Definisi model & bobot tersimpan
│   ├── cbam_resnet.py      # Implementasi PyTorch CBAM-ResNet
│   └── resnet_cbam.pth     # Bobot model terlatih (Best Model)
├── notebooks/              # Jupyter Notebooks untuk riset & eksperimen
│   └── cbam_resnet_research.ipynb
├── src/                    # Source code pelatihan & utilitas
│   ├── train.py            # Skrip pelatihan utama
│   └── prepare_data.py     # Skrip persiapan data
├── streamlit_app.py        # Aplikasi Web Streamlit (Inference Demo)
├── requirements.txt        # Dependensi Python
└── README.md               # Dokumentasi Proyek
```

---

## 🚀 Memulai (Getting Started)

### Prasyarat
*   Python 3.10 atau lebih baru
*   CUDA (Opsional, disarankan untuk pelatihan)

### Instalasi

Proyek ini menggunakan `uv` untuk manajemen paket yang cepat, namun juga mendukung `pip` standar.

**Opsi 1: Menggunakan `uv` (Disarankan)**
```bash
# Instal uv jika belum ada
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sinkronisasi dependensi
uv sync
```

**Opsi 2: Menggunakan `pip`**
```bash
pip install -r requirements.txt
```

---

## 💻 Penggunaan

### 1. Persiapan Data
Pastikan dataset Anda terstruktur dalam format `ImageFolder` standar:
```
data/
  train/
    organic/
    recyclable/
  val/
    organic/
    recyclable/
```

### 2. Pelatihan Model
Jalankan skrip pelatihan untuk memulai proses training dari awal:
```bash
uv run python src/train.py
# atau
python src/train.py
```
Log pelatihan akan disimpan di `training.log` dan model terbaik akan disimpan di `models/resnet_cbam.pth`.

### 3. Menjalankan Aplikasi Demo
Gunakan Streamlit untuk mencoba model secara interaktif melalui browser:
```bash
uv run streamlit run streamlit_app.py
# atau
streamlit run streamlit_app.py
```
Akses aplikasi di `http://localhost:8501`.

---

## 🧠 Arsitektur Model

Model ini menggunakan pendekatan **Residual Attention Network**:
1.  **Backbone**: ResNet-34 (34 Layer Convolutional).
2.  **Attention**: Modul CBAM disisipkan **di dalam** setiap `BasicBlock` pada jalur residual.
3.  **Mekanisme**:
    *   *Channel Attention*: Menilai "apa" yang penting (fitur konten).
    *   *Spatial Attention*: Menilai "di mana" fitur penting berada (lokalisasi).

Detail lengkap arsitektur dapat dilihat di [Dokumentasi Arsitektur](docs/architecture_details.md).

---

## 📊 Hasil & Performa

Berdasarkan eksperimen pada dataset uji:

| Metrik | Nilai |
| :--- | :--- |
| **Akurasi Validasi** | **94.5%** |
| **Precision (Avg)** | 92% |
| **Recall (Avg)** | 92% |
| **F1-Score** | 0.92 |

---

## 📚 Dokumentasi & Publikasi

*   **[Draft Paper Riset](docs/paper_draft.md)**: Penjelasan mendalam tentang metodologi, eksperimen, dan analisis hasil (Standar Scopus).
*   **[Detail Arsitektur](docs/architecture_details.md)**: Spesifikasi teknis layer-by-layer dan diagram alur.

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan buat *Issue* untuk diskusi fitur atau *Pull Request* untuk perbaikan kode.

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---
**Dikembangkan oleh EGGI SATRIA, DAFFA AHMAD, ANISSA FItriani**
*2025*