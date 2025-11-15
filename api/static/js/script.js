// script.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultSection = document.getElementById('resultSection');
    const loading = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const predictionEl = document.getElementById('prediction');
    const confidenceEl = document.getElementById('confidence');
    const errorMessageEl = document.getElementById('errorMessage');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const fileInput = document.getElementById('imageInput');
        const file = fileInput.files[0];

        if (!file) {
            showError('Please select an image file.');
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            showError('Please select a valid image file.');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size must be less than 10MB.');
            return;
        }

        // Show loading
        showLoading();

        // Create FormData
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const result = await response.json();
            showResult(result);

        } catch (error) {
            showError(error.message);
        }
    });

    function showLoading() {
        resultSection.style.display = 'none';
        errorDiv.style.display = 'none';
        loading.style.display = 'block';
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyzing...';
    }

    function showResult(result) {
        loading.style.display = 'none';
        errorDiv.style.display = 'none';
        resultSection.style.display = 'block';
        predictBtn.disabled = false;
        predictBtn.textContent = 'Classify Waste';

        predictionEl.textContent = `Prediction: ${result.prediction}`;
        confidenceEl.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
    }

    function showError(message) {
        loading.style.display = 'none';
        resultSection.style.display = 'none';
        errorDiv.style.display = 'block';
        predictBtn.disabled = false;
        predictBtn.textContent = 'Classify Waste';

        errorMessageEl.textContent = message;
    }

    // File input change handler
    document.getElementById('imageInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const label = e.target.nextElementSibling;
            label.textContent = file.name;
        }
    });
});