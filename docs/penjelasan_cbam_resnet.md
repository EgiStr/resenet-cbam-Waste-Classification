# Penjelasan CBAM-ResNet untuk Klasifikasi Sampah

## Pendahuluan

Dalam penelitian ini, kami memilih arsitektur CBAM-ResNet sebagai model utama untuk klasifikasi sampah. CBAM (Convolutional Block Attention Module) dikombinasikan dengan ResNet (Residual Network) memberikan kemampuan yang unggul dalam mengenali pola visual pada gambar sampah dengan berbagai kategori. Pendekatan ini didasarkan pada penelitian terkini yang menunjukkan efektivitas attention mechanism dalam tugas klasifikasi visual.

## Alasan Pemilihan Model

### 1. Keunggulan CBAM (Convolutional Block Attention Module)
CBAM adalah modul attention yang dirancang khusus untuk jaringan konvolusi. Modul ini bekerja dengan cara:
- **Channel Attention**: Memberikan bobot yang berbeda pada setiap channel fitur untuk menekankan informasi yang relevan
- **Spatial Attention**: Fokus pada lokasi spasial yang penting dalam gambar

Menurut penelitian Woo et al. (2018) dalam paper "CBAM: Convolutional Block Attention Module" (arXiv:1807.06521), CBAM dapat meningkatkan performa model klasifikasi tanpa menambah kompleksitas komputasi yang signifikan.

### 2. Keunggulan ResNet
ResNet menggunakan residual connections yang memungkinkan:
- Pelatihan jaringan yang lebih dalam tanpa masalah vanishing gradient
- Representasi fitur yang lebih kaya untuk tugas klasifikasi kompleks
- Konvergensi yang lebih stabil selama pelatihan

### 3. Kombinasi CBAM-ResNet untuk Klasifikasi Sampah
Untuk klasifikasi sampah, kombinasi ini memberikan keuntungan:
- **Attention Mechanism**: Dapat fokus pada bagian penting dari gambar sampah (seperti bentuk, warna, tekstur)
- **Deep Feature Learning**: ResNet dapat mengekstrak fitur hierarkis dari gambar sampah
- **Robustness**: Model dapat menangani variasi pencahayaan, sudut, dan kondisi gambar sampah di dunia nyata

## Penjelasan Komponen Teknis

### Arsitektur CBAM
```
Input Feature Map → Channel Attention → Spatial Attention → Output Feature Map
```

**Channel Attention Module:**
- Squeeze: Global Average Pooling dan Global Max Pooling
- Excitation: Fully Connected layers dengan ReLU dan Sigmoid
- Scale: Element-wise multiplication dengan input

**Spatial Attention Module:**
- Average Pooling dan Max Pooling di channel dimension
- Convolution 7x7 untuk menghasilkan spatial attention map
- Sigmoid activation

### Arsitektur ResNet dengan CBAM
Model menggunakan ResNet-34 sebagai backbone dengan CBAM diintegrasikan pada setiap residual block:

```
Conv1 (7x7, stride 2) → MaxPool → Residual Block 1-3 → Residual Block 4-6 → Residual Block 7-9 → Average Pool → FC
```

Setiap residual block terdiri dari:
- Dua convolution layers 3x3
- Batch Normalization
- ReLU activation
- CBAM module setelah setiap block

## Penjelasan Arsitektur Layer by Layer

### 1. Input Layer
- **Ukuran Input**: 224 × 224 × 3 (RGB)
- **Preprocessing**: Resize, center crop, normalization dengan ImageNet mean/std
- **Fungsi**: Menerima gambar sampah yang telah dipreprocessing

### 2. Convolutional Layer 1 (Conv1)
- **Tipe Layer**: Conv2d
- **Parameter**: kernel_size=7, stride=2, padding=3, out_channels=64, bias=False
- **Input Size**: 224 × 224 × 3
- **Output Size**: 112 × 112 × 64
- **Operasi**: Konvolusi 7×7 dengan stride 2 untuk ekstraksi fitur awal
- **Batch Normalization**: Ya (64 channels)
- **Activation**: ReLU

### 3. Max Pooling Layer
- **Tipe Layer**: MaxPool2d
- **Parameter**: kernel_size=3, stride=2, padding=1
- **Input Size**: 112 × 112 × 64
- **Output Size**: 56 × 56 × 64
- **Fungsi**: Reduksi dimensi spasial dan invariance terhadap translasi kecil

### 4. Residual Layer 1 (Layer1)
- **Jumlah Block**: 3 BasicBlock + 3 CBAM modules
- **Channels**: 64
- **Input Size**: 56 × 56 × 64
- **Output Size**: 56 × 56 × 64 (stride=1, tidak ada downsampling)

**Setiap BasicBlock dalam Layer1:**
- **Conv2d_1**: kernel_size=3, stride=1, padding=1, in_channels=64, out_channels=64
- **BatchNorm2d_1**: 64 channels
- **ReLU_1**: Activation
- **Conv2d_2**: kernel_size=3, stride=1, padding=1, in_channels=64, out_channels=64
- **BatchNorm2d_2**: 64 channels
- **Shortcut**: Identity (karena stride=1 dan channels sama)
- **ReLU_2**: Final activation
- **CBAM Module**: Channel + Spatial Attention

### 5. Residual Layer 2 (Layer2)
- **Jumlah Block**: 4 BasicBlock + 4 CBAM modules
- **Channels**: 128
- **Input Size**: 56 × 56 × 64
- **Output Size**: 28 × 28 × 128 (stride=2 pada block pertama)

**Block Pertama (stride=2):**
- **Conv2d_1**: kernel_size=3, stride=2, padding=1, in_channels=64, out_channels=128
- **BatchNorm2d_1**: 128 channels
- **ReLU_1**: Activation
- **Conv2d_2**: kernel_size=3, stride=1, padding=1, in_channels=128, out_channels=128
- **BatchNorm2d_2**: 128 channels
- **Shortcut**: Conv2d(1×1, stride=2) + BatchNorm2d untuk downsampling
- **ReLU_2**: Final activation
- **CBAM Module**: Channel + Spatial Attention

**Block 2-4 (stride=1):**
- Sama seperti Layer1 tapi dengan 128 channels

### 6. Residual Layer 3 (Layer3)
- **Jumlah Block**: 6 BasicBlock + 6 CBAM modules
- **Channels**: 256
- **Input Size**: 28 × 28 × 128
- **Output Size**: 14 × 14 × 256 (stride=2 pada block pertama)

**Struktur sama seperti Layer2, hanya channels berubah ke 256**

### 7. Residual Layer 4 (Layer4)
- **Jumlah Block**: 3 BasicBlock + 3 CBAM modules
- **Channels**: 512
- **Input Size**: 14 × 14 × 256
- **Output Size**: 7 × 7 × 512 (stride=2 pada block pertama)

**Struktur sama seperti Layer2, hanya channels berubah ke 512**

### 8. Global Average Pooling
- **Tipe Layer**: AdaptiveAvgPool2d
- **Parameter**: output_size=(1, 1)
- **Input Size**: 7 × 7 × 512
- **Output Size**: 1 × 1 × 512
- **Fungsi**: Global spatial pooling untuk mengubah feature map menjadi vektor fitur

### 9. Flatten Layer
- **Input Size**: 1 × 1 × 512
- **Output Size**: 512 (vektor 1D)
- **Fungsi**: Mengubah tensor 4D menjadi vektor untuk input ke fully connected layer

### 10. Fully Connected Layer (FC)
- **Tipe Layer**: Linear
- **Parameter**: in_features=512, out_features=num_classes (2 untuk binary classification)
- **Input Size**: 512
- **Output Size**: 2 (logits untuk setiap kelas)
- **Fungsi**: Klasifikasi akhir berdasarkan fitur yang diekstrak
- **Activation**: Softmax (di luar model, selama inference)

## Detail CBAM Module

### Channel Attention Branch
1. **AdaptiveAvgPool2d**: Input → 1×1×C
2. **Conv2d**: 1×1 conv, C → C//16 channels, ReLU
3. **Conv2d**: 1×1 conv, C//16 → C channels
4. **AdaptiveMaxPool2d**: Input → 1×1×C (parallel)
5. **Conv2d**: 1×1 conv, C → C//16 channels, ReLU (parallel)
6. **Conv2d**: 1×1 conv, C//16 → C channels (parallel)
7. **Addition**: Avg + Max outputs
8. **Sigmoid**: Generate channel attention weights

### Spatial Attention Branch
1. **Global Average Pooling**: Across channels → H×W×1
2. **Global Max Pooling**: Across channels → H×W×1 (parallel)
3. **Concatenation**: [Avg, Max] → H×W×2
4. **Conv2d**: 7×7 conv, 2 → 1 channels, padding=3
5. **Sigmoid**: Generate spatial attention map

## Ringkasan Parameter Model

- **Total Parameters**: ~21.3 juta (tergantung num_classes)
- **Trainable Parameters**: Semua layer kecuali mungkin beberapa BatchNorm stats
- **Memory Footprint**: ~81 MB untuk float32
- **FLOPs**: ~3.6 GFLOPs per forward pass (224×224 input)

## Alur Forward Pass

```
Input (224×224×3) 
    ↓
Conv1 + BN + ReLU (112×112×64)
    ↓
MaxPool (56×56×64)
    ↓
Layer1: 3×[BasicBlock + CBAM] (56×56×64)
    ↓
Layer2: 4×[BasicBlock + CBAM] (28×28×128)
    ↓
Layer3: 6×[BasicBlock + CBAM] (14×14×256)
    ↓
Layer4: 3×[BasicBlock + CBAM] (7×7×512)
    ↓
AdaptiveAvgPool (1×1×512)
    ↓
Flatten (512)
    ↓
FC Layer (2)
    ↓
Output: Class logits
```

## Input dan Output

### Input
- **Ukuran Gambar**: 224x224x3 (RGB)
- **Preprocessing**: Resize, normalization dengan ImageNet mean/std
- **Data Augmentation**: Random crop, horizontal flip, color jitter

### Output
- **Jumlah Kelas**: Bervariasi tergantung dataset (biasanya 6-12 kategori sampah)
- **Activation**: Softmax untuk klasifikasi multi-kelas
- **Loss Function**: Cross-Entropy Loss

## Fungsi Aktivasi

### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
```
Digunakan pada semua layer konvolusi untuk:
- Mengatasi vanishing gradient
- Mempercepat konvergensi
- Memberikan sparsity pada aktivasi

### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
```
Digunakan pada attention modules untuk menghasilkan bobot antara 0-1.

## Penelitian Pendukung

### 1. CBAM: Convolutional Block Attention Module
- **Penulis**: Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
- **Sumber**: arXiv:1807.06521
- **Kontribusi**: Memperkenalkan CBAM yang meningkatkan performa model CNN tanpa overhead komputasi besar
- **Relevansi**: Attention mechanism sangat efektif untuk klasifikasi sampah karena dapat fokus pada fitur visual yang membedakan kategori sampah

### 2. TACO: Trash Annotations in Context for Litter Detection
- **Penulis**: Pedro F. Proença, Pedro Simões
- **Sumber**: arXiv:2003.06975
- **Kontribusi**: Dataset TACO untuk deteksi sampah dengan 1500 gambar dan 4784 anotasi
- **Relevansi**: Menunjukkan tantangan dan pendekatan untuk klasifikasi sampah menggunakan CNN

### 3. Penelitian Terkait Klasifikasi Sampah
Berdasarkan kajian literatur, CNN telah terbukti efektif untuk klasifikasi sampah karena kemampuannya mengekstrak fitur visual kompleks. CBAM meningkatkan kemampuan ini dengan memberikan attention pada fitur yang relevan.

## Kesimpulan

CBAM-ResNet dipilih karena kombinasi attention mechanism dan deep residual learning yang ideal untuk tugas klasifikasi sampah. Model ini dapat menangani kompleksitas visual gambar sampah dengan baik, memberikan performa yang superior dibandingkan model CNN konvensional. Penelitian pendukung menunjukkan bahwa pendekatan ini state-of-the-art untuk aplikasi klasifikasi sampah berbasis computer vision.