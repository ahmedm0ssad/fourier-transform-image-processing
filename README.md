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

## 📁 Project Structure

```
Morphology/
├── Forrier transform.ipynb    # Main Jupyter notebook with implementations
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

4. **Open `Forrier transform.ipynb`** and run the cells sequentially.

## 📊 Usage

### Basic Fourier Transform
```python
# Load and process an image
image = load_image(image_url)
dft_shifted = apply_fourier_transform(image)
```

### Apply Frequency Filters
```python
# Create and apply a low-pass filter
lowpass_filter = create_filter(image.shape, 'lowpass', d0=50)
filtered_image = inverse_fourier_transform(dft_shifted * lowpass_filter)
```

### Compare Filter Types
```python
# Compare Ideal, Butterworth, and Gaussian filters
filter_types = ['ideal', 'butterworth', 'gaussian']
for filter_type in filter_types:
    filter_mask = create_filter(image.shape, filter_type, cut_off_frequency=50)
    # Process and display results
```

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
