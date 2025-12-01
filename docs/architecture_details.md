# Spesifikasi Detail Arsitektur CBAM-ResNet34

Dokumen ini menjelaskan secara rinci arsitektur *Deep Learning* yang digunakan dalam proyek klasifikasi sampah, yaitu **ResNet-34** yang diintegrasikan dengan **Convolutional Block Attention Module (CBAM)**.

## 1. Tinjauan Umum Arsitektur

Model ini dirancang untuk menerima citra input RGB berukuran $224 \times 224$ piksel dan mengklasifikasikannya ke dalam dua kelas: **Organik** dan **Daur Ulang**. Arsitektur dasar ResNet-34 dipilih karena keseimbangannya antara kedalaman jaringan (kemampuan representasi) dan efisiensi komputasi. Penambahan CBAM bertujuan untuk meningkatkan kemampuan model dalam memfokuskan atensi pada fitur-fitur relevan (objek sampah) dan menekan *noise* dari latar belakang.

### Diagram Alur Tingkat Tinggi
```mermaid
graph TD
    Input[Input Image 224x224x3] --> Stem[Initial Conv & MaxPool]
    Stem --> L1[Layer 1: 3x (BasicBlock + CBAM)]
    L1 --> L2[Layer 2: 4x (BasicBlock + CBAM)]
    L2 --> L3[Layer 3: 6x (BasicBlock + CBAM)]
    L3 --> L4[Layer 4: 3x (BasicBlock + CBAM)]
    L4 --> GAP[Global Average Pooling]
    GAP --> FC[Fully Connected Layer]
    FC --> Output[Softmax Probabilities]
```

---

## 2. Komponen Utama

### 2.1 Backbone: ResNet-34
ResNet-34 menggunakan **BasicBlock** sebagai unit pembangun utamanya. Setiap BasicBlock terdiri dari dua lapisan konvolusi $3 \times 3$ dengan koneksi residual (*skip connection*).

**Struktur BasicBlock:**
1.  Conv2d ($3 \times 3$, padding=1)
2.  Batch Normalization
3.  ReLU Activation
4.  Conv2d ($3 \times 3$, padding=1)
5.  Batch Normalization
6.  *Skip Connection* (Penjumlahan dengan input blok)
7.  ReLU Activation

### 2.2 Modul Atensi: CBAM
CBAM adalah modul atensi ringan yang terdiri dari dua sub-modul yang dijalankan secara berurutan: **Channel Attention Module (CAM)** dan **Spatial Attention Module (SAM)**.

#### A. Channel Attention Module (CAM)
Fokus pada **"apa"** fitur yang penting (misal: tekstur plastik vs daun).
*   **Input**: Feature map $F \in \mathbb{R}^{C \times H \times W}$
*   **Operasi**:
    1.  **Global Average Pooling**: Menghasilkan deskriptor spasial rata-rata ($F_{avg}^c$).
    2.  **Global Max Pooling**: Menghasilkan deskriptor spasial puncak ($F_{max}^c$).
    3.  **Shared MLP**: Kedua deskriptor diproses oleh *Multi-Layer Perceptron* (MLP) yang sama dengan satu *hidden layer*. Rasio reduksi ($r$) diatur ke 16.
        *   $W_0 \in \mathbb{R}^{C/r \times C}$, $W_1 \in \mathbb{R}^{C \times C/r}$
    4.  **Penjumlahan & Aktivasi**: Output MLP dijumlahkan dan diproses fungsi Sigmoid.
*   **Rumus Matematis**:
    $$M_c(F) = \sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))$$
    $$M_c(F) = \sigma(W_1(W_0(F_{avg}^c)) + W_1(W_0(F_{max}^c)))$$

#### B. Spatial Attention Module (SAM)
Fokus pada **"di mana"** fitur penting berada (lokalisasi objek).
*   **Input**: Feature map yang telah diperhalus oleh CAM, $F' = M_c(F) \otimes F$.
*   **Operasi**:
    1.  **Channel-wise Pooling**: Melakukan Average Pooling dan Max Pooling sepanjang sumbu channel, menghasilkan dua peta fitur 2D ($H \times W$).
    2.  **Concatenation**: Menggabungkan kedua peta fitur tersebut.
    3.  **Convolution**: Lapisan konvolusi $7 \times 7$ untuk mengagregasi informasi spasial.
    4.  **Aktivasi**: Fungsi Sigmoid.
*   **Rumus Matematis**:
    $$M_s(F') = \sigma(f^{7 \times 7}([AvgPool(F'); MaxPool(F')]))$$

---

## 3. Integrasi dan Konfigurasi Layer

Dalam implementasi ini, modul CBAM disisipkan **di dalam** *BasicBlock*, tepatnya pada jalur residual (*residual path*) sebelum penjumlahan dengan *shortcut connection*.

**Urutan Pemrosesan per Blok:**
`Input` $\rightarrow$ `Conv1` $\rightarrow$ `BN` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2` $\rightarrow$ `BN` $\rightarrow$ **`CBAM`** $\rightarrow$ `Add Shortcut` $\rightarrow$ `ReLU`

### Tabel Spesifikasi Layer Detail

| Nama Layer | Output Size | Struktur Detail | Parameter Kunci |
| :--- | :--- | :--- | :--- |
| **Input** | $224 \times 224 \times 3$ | - | - |
| **Initial Conv** | $112 \times 112 \times 64$ | Conv $7 \times 7$, stride 2, padding 3<br>Batch Norm, ReLU | Kernel=7, Stride=2 |
| **Max Pool** | $56 \times 56 \times 64$ | MaxPool $3 \times 3$, stride 2, padding 1 | Kernel=3, Stride=2 |
| **Layer 1** | $56 \times 56 \times 64$ | $\begin{bmatrix} \text{BasicBlock} \\ \text{CBAM} \end{bmatrix} \times 3$ | Stride=1, Channels=64 |
| **Layer 2** | $28 \times 28 \times 128$ | $\begin{bmatrix} \text{BasicBlock} \\ \text{CBAM} \end{bmatrix} \times 4$ | Stride=2 (di blok pertama), Channels=128 |
| **Layer 3** | $14 \times 14 \times 256$ | $\begin{bmatrix} \text{BasicBlock} \\ \text{CBAM} \end{bmatrix} \times 6$ | Stride=2 (di blok pertama), Channels=256 |
| **Layer 4** | $7 \times 7 \times 512$ | $\begin{bmatrix} \text{BasicBlock} \\ \text{CBAM} \end{bmatrix} \times 3$ | Stride=2 (di blok pertama), Channels=512 |
| **Global Pool** | $1 \times 1 \times 512$ | Adaptive Average Pooling | Output size=(1,1) |
| **Flatten** | 512 | Flatten tensor | - |
| **FC Layer** | 2 | Linear(512, 2) | Num Classes=2 |

### Total Parameter
*   **Backbone (ResNet-34)**: ~21.8 Juta parameter
*   **Tambahan CBAM**: Sangat ringan (~0.1 - 0.5% penambahan parameter)
*   **Total**: ~22 Juta parameter

## 4. Implementasi Kode (Snippet)

Berikut adalah representasi kode dari integrasi blok tersebut (berdasarkan `models/cbam_resnet.py`):

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(BasicBlock, self).__init__()
        # ... (conv1, bn1, relu, conv2, bn2 definitions) ...
        self.cbam = CBAM(out_channels) if use_cbam else None
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.cbam:
            out = self.cbam(out) # CBAM applied to residual features

        out += self.shortcut(residual)
        out = self.relu(out)
        return out
```

Konfigurasi ini memastikan bahwa setiap fitur yang diekstraksi oleh blok residual langsung divalidasi dan diperhalus oleh mekanisme atensi sebelum digabungkan dengan identitas asli, memaksimalkan efektivitas propagasi fitur yang relevan.
