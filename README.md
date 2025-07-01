# Fourier Transform Image Processing

A comprehensive implementation of Fourier Transform techniques for image processing using Python, OpenCV, and NumPy.

## 📋 Overview

This project demonstrates various applications of the Fourier Transform in digital image processing, including frequency domain filtering, different filter types, and their effects on image content. The implementation covers both theoretical concepts and practical applications.

## 🚀 Features

- **Discrete Fourier Transform (DFT)** implementation for images
- **Frequency Domain Filtering**:
  - Low-pass filters (smoothing)
  - High-pass filters (edge enhancement)
  - Band-pass filters (selective frequency isolation)
  - Band-stop filters (frequency notch filtering)
- **Multiple Filter Types**:
  - Ideal filters
  - Butterworth filters
  - Gaussian filters
- **Interactive visualizations** using Matplotlib
- **Comprehensive comparisons** between different filtering approaches
- **Dual Implementation**:
  - Interactive Jupyter notebook for learning and experimentation
  - Standalone Python module (`FourierImageProcessor` class) for integration
- **Professional Documentation** with type hints and comprehensive docstrings

## 📁 Project Structure

```
Morphology/
├── Forrier transform.ipynb    # Interactive Jupyter notebook with step-by-step implementation
├── fourier_transform.py       # Standalone Python module with FourierImageProcessor class
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── LICENSE                   # MIT License
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Morphology
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open and run the project**:
   - **Jupyter Notebook**: `jupyter notebook` → Open `Forrier transform.ipynb`
   - **Python Script**: `python fourier_transform.py`

## 📊 Usage

### Option 1: Interactive Jupyter Notebook
Open `Forrier transform.ipynb` for step-by-step interactive exploration with detailed explanations.

### Option 2: Python Module
Use the `FourierImageProcessor` class for programmatic access:

```python
from fourier_transform import FourierImageProcessor

# Initialize processor
processor = FourierImageProcessor()

# Load and process an image
image = processor.load_image("image_url_or_path")
dft_shifted = processor.apply_fourier_transform(image)
```

### Basic Fourier Transform
```python
# Load and process an image
image = processor.load_image(image_url)
dft_shifted = processor.apply_fourier_transform(image)

# Get magnitude spectrum for visualization
magnitude_spectrum = processor.get_magnitude_spectrum(dft_shifted)
processor.show_image(magnitude_spectrum, "Magnitude Spectrum")
```

### Apply Frequency Filters
```python
# Create and apply a low-pass filter
filtered_image, filter_mask = processor.apply_frequency_filter(
    image, 'gaussian', cut_off_frequency=50, highpass=False
)

# Apply high-pass filter
filtered_image, filter_mask = processor.apply_frequency_filter(
    image, 'butterworth', cut_off_frequency=30, highpass=True, order=2
)
```

### Compare Filter Types
```python
# Compare Ideal, Butterworth, and Gaussian filters
filter_types = ['ideal', 'butterworth', 'gaussian']
results = processor.compare_filters(image, filter_types, cut_off_frequency=50)
processor.plot_comparison(image, results, "Low-pass Filters")
```

## 🐍 Python Module Features

The `fourier_transform.py` module provides a complete `FourierImageProcessor` class with:

### Core Methods:
- `load_image(url)` - Load images from URLs
- `load_local_image(path)` - Load images from local paths
- `apply_fourier_transform(image)` - Compute DFT with frequency shifting
- `inverse_fourier_transform(dft)` - Reconstruct images from frequency domain
- `get_magnitude_spectrum(dft)` - Visualize frequency content

### Filter Creation:
- `create_filter(shape, type, cutoff, **kwargs)` - Generate various filter types
- `apply_frequency_filter(image, type, cutoff)` - Apply filters in one step
- `compare_filters(image, types, cutoff)` - Compare multiple filter types

### Visualization:
- `show_image(image, title)` - Display images with matplotlib
- `plot_comparison(image, results, prefix)` - Side-by-side filter comparisons

### Supported Filter Types:
- **Ideal**: Sharp cutoff (may cause ringing)
- **Butterworth**: Smooth transition with configurable order
- **Gaussian**: Smoothest transition, no ringing artifacts
- **Band-pass/Band-stop**: Frequency range isolation/removal

## 🔍 Key Concepts Covered

### 1. Fourier Transform Fundamentals
- Spatial to frequency domain conversion
- Magnitude and phase spectrum analysis
- Zero-frequency component shifting

### 2. Frequency Domain Filtering
- **Low-pass filtering**: Removes high-frequency noise, smooths images
- **High-pass filtering**: Enhances edges and fine details
- **Band-pass filtering**: Isolates specific frequency ranges
- **Band-stop filtering**: Removes specific frequency ranges

### 3. Filter Characteristics
- **Ideal Filters**: Sharp cutoff, may cause ringing artifacts
- **Butterworth Filters**: Smooth transition, good compromise
- **Gaussian Filters**: Smoothest transition, no ringing

## 📈 Results and Applications

The notebook demonstrates how different filters affect image content:

- **Smoothing effects** of low-pass filters for noise reduction
- **Edge enhancement** capabilities of high-pass filters
- **Selective frequency manipulation** using band-pass/band-stop filters
- **Comparative analysis** of filter types and their trade-offs

## 🔧 Technical Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- Matplotlib
- Jupyter Notebook
- IPython

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Educational Context

This project is part of the **Computer Vision DSAI 352** course curriculum, specifically focusing on:
- Morphological operations in image processing
- Frequency domain analysis
- Digital image filtering techniques
- Practical applications of mathematical transforms

## 🔬 Further Applications

The techniques demonstrated in this project can be extended to:
- **Medical Image Processing**: MRI, CT scan enhancement
- **Remote Sensing**: Satellite image analysis
- **Quality Control**: Industrial inspection systems
- **Artistic Effects**: Creative image manipulation
- **Signal Processing**: 1D signal filtering applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Computer Vision DSAI 352 Assignment**
- Course: Computer Vision and Digital Image Processing
- Institution: ZC-UST
- Academic Year: 3rd Year, Semester 2

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- OpenCV and NumPy development teams
- Jupyter Project for the interactive development environment
- Scientific Python community for excellent documentation and examples

## 📧 Contact

**Ahmed Mossad** - [ahmed.abdelfattah.mossad@gmail.com](mailto:ahmed.abdelfattah.mossad@gmail.com)

Project Link: [https://github.com/ahmedm0ssad/SIFT-Texture-Image-Retrieval](https://github.com/ahmedm0ssad/SIFT-Texture-Image-Retrieval)

---



**Note**: This project is developed for educational purposes as part of a computer vision course curriculum. The implementations are designed to demonstrate fundamental concepts and may be optimized further for production use.
