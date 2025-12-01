# Analisis Proses Training CBAM-ResNet Waste Classification

## 1. Proses Training yang Telah Dilakukan

### Pembagian Dataset
Berdasarkan konfigurasi dalam notebook, dataset dibagi menjadi tiga bagian utama:
- **Training set**: Digunakan untuk melatih model
- **Validation set**: Digunakan untuk monitoring performa selama training dan hyperparameter tuning
- **Test set**: Digunakan untuk evaluasi akhir model

Dataset disimpan dalam struktur direktori terpisah (`train/`, `val/`, `test/`) dengan dua kelas utama: 'organic' dan 'recyclable'.

### Hyperparameter Awal
Hyperparameter awal yang dikonfigurasi dalam sistem:
- **Epochs**: 50
- **Learning Rate**: 1e-3 (0.001)
- **Weight Decay**: 1e-4 (0.0001)
- **Momentum**: 0.9
- **Batch Size**: 32
- **Image Size**: 224x224 pixels
- **Optimizer**: AdamW
- **Scheduler**: StepLR dengan step_size dan gamma
- **Loss Function**: CrossEntropyLoss

### Durasi Training
Durasi training tidak secara eksplisit disebutkan dalam notebook karena belum dijalankan, namun sistem mencatat:
- Waktu per epoch (epoch_time)
- Total training time
- Menggunakan mixed precision training untuk mempercepat proses

### Kendala yang Dihadapi
Berdasarkan kode yang ada, kendala potensial yang dapat muncul:
- **Komputasi**: Training dengan batch size 32 dan 50 epochs membutuhkan resource GPU yang cukup
- **Memory**: Model CBAM-ResNet dengan attention mechanism membutuhkan lebih banyak memory
- **Data Imbalance**: Tidak ada explicit handling untuk class imbalance dalam kode
- **Overfitting**: Dengan 50 epochs, risiko overfitting tinggi tanpa early stopping yang ketat

## 2. Teknik Tuning yang Dilakukan

### Teknik Tuning
Notebook mengimplementasikan dua teknik utama hyperparameter tuning:

1. **Grid Search**: Menggunakan `ParameterGrid` dari scikit-learn untuk exhaustive search
2. **Bayesian Optimization**: Menggunakan Optuna dengan TPE (Tree-structured Parzen Estimator) sampler

### Alasan Pemilihan Teknik
- **Grid Search**: Dipilih karena exhaustive dan deterministik, memungkinkan eksplorasi sistematis parameter space
- **Bayesian Optimization**: Dipilih karena lebih efisien untuk high-dimensional parameter space dan dapat menemukan optimum dengan trial yang lebih sedikit dibanding random search

## 3. Variasi Hyperparameter yang Dicobakan

### Parameter yang Dituning
Variasi hyperparameter yang diuji dalam kedua metode tuning:

- **Learning Rate (lr)**: 1e-5 sampai 1e-2 (log scale)
- **Weight Decay**: 1e-6 sampai 1e-3 (log scale)  
- **Reduction Ratio**: [4, 8, 16, 32]
- **Kernel Size**: [3, 5, 7]
- **Batch Size**: [16, 32, 64]

### Parameter CBAM-Specific
- **reduction_ratio**: Mengontrol kompresi channel attention
- **kernel_size**: Ukuran kernel untuk spatial attention

## 4. Hasil Sementara Tuning

### Status Tuning
**Belum ada hasil tuning yang tercatat** karena notebook belum dieksekusi. Kode menyediakan framework untuk:
- Menjalankan grid search dengan maksimal 10 epochs per kombinasi
- Bayesian optimization dengan 20 trials
- Plotting hasil tuning

### Performance Terbaik yang Diharapkan
Berdasarkan konfigurasi, target performance:
- **Validation Accuracy**: > 90% (dengan CBAM enhancement)
- **F1-Score**: > 0.85 untuk kedua kelas
- **Training Stability**: Konvergen dalam 50 epochs tanpa overfitting signifikan

## 5. Grafik Training vs Validation

### Grafik yang Tersedia
Notebook menyediakan plotting untuk:
- **Training Loss vs Validation Loss**
- **Training Accuracy vs Validation Accuracy**  
- **Training F1 vs Validation F1**
- **Learning Rate Schedule**
- **Epoch Time**

### Contoh Visualisasi
```python
# Plot training history
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.subplot(2, 2, 2)  
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Dan seterusnya untuk F1 dan learning rate
```

## 6. Proses Evaluasi

### Metrik Evaluasi yang Dipilih
Sistem menggunakan comprehensive evaluation metrics:

**Overall Metrics:**
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted)  
- F1-Score (macro & weighted)

**Per-Class Metrics:**
- Precision, Recall, F1 untuk setiap kelas ('organic', 'recyclable')

**Advanced Metrics:**
- Confusion Matrix
- ROC Curve & AUC (untuk binary classification)
- Classification Report

### Hasil Evaluasi Sementara
**Belum ada hasil evaluasi** karena training belum dijalankan. Framework siap untuk mengukur:

- **Accuracy**: > 85% target untuk waste classification
- **F1-Score**: Balanced metric untuk imbalanced classes
- **Per-Class Performance**: Memastikan kedua kelas (organic/recyclable) memiliki performa baik

### Tren Awal yang Diharapkan
Berdasarkan konfigurasi model:
- **Konvergensi**: Model diharapkan konvergen dalam 20-30 epochs
- **Stability**: CBAM attention mechanism membantu stability
- **Generalization**: Validation metrics tidak menunjukkan overfitting signifikan
- **Class Balance**: F1-score macro vs weighted menunjukkan class distribution

### Performance Analysis Tambahan
- **Inference Time**: Diukur dalam milliseconds
- **Throughput**: Samples per second
- **Memory Usage**: GPU memory consumption

## Kesimpulan

Notebook menyediakan framework komprehensif untuk training CBAM-ResNet dengan:
- Sistematic hyperparameter tuning (Grid Search + Bayesian Optimization)
- Comprehensive evaluation metrics
- Visualization tools untuk monitoring training
- Baseline comparison dengan model standar
- Ablation study untuk CBAM components

Namun, **semua hasil masih dalam tahap implementasi** karena notebook belum dieksekusi. Untuk mendapatkan hasil aktual, perlu menjalankan training dengan dataset yang sesuai.</content>
<parameter name="filePath">/mnt/2A28ACA028AC6C8F/Programming/dataScience/resenet-cbam/training_analysis.md