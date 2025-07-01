"""
Fourier Transform Image Processing

A comprehensive implementation of Fourier Transform techniques for image processing
using Python, OpenCV, and NumPy.

Author: Ahmed Mossad
Course: Computer Vision DSAI 352
Institution: ZC-UST
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from typing import Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class FourierImageProcessor:
    """
    A class for applying Fourier Transform operations on images.
    """
    
    def __init__(self):
        """Initialize the Fourier Image Processor."""
        pass
    
    @staticmethod
    def load_image(image_url: str) -> np.ndarray:
        """
        Load image from URL in grayscale.
        
        Args:
            image_url (str): URL of the image to load
            
        Returns:
            np.ndarray: Grayscale image array
        """
        try:
            resp = urllib.request.urlopen(image_url)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def load_local_image(image_path: str) -> np.ndarray:
        """
        Load image from local path in grayscale.
        
        Args:
            image_path (str): Local path to the image
            
        Returns:
            np.ndarray: Grayscale image array
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return image
        except Exception as e:
            print(f"Error loading local image: {e}")
            return None
    
    @staticmethod
    def show_image(image: np.ndarray, title: str = "Image", cmap: str = 'gray', 
                   figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Display image using matplotlib.
        
        Args:
            image (np.ndarray): Image to display
            title (str): Title for the image
            cmap (str): Colormap for display
            figsize (tuple): Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def apply_fourier_transform(image: np.ndarray) -> np.ndarray:
        """
        Compute the DFT and shift the zero-frequency component to the center.
        
        Args:
            image (np.ndarray): Input grayscale image
            
        Returns:
            np.ndarray: DFT shifted to center
        """
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)
        return dft_shifted
    
    @staticmethod
    def inverse_fourier_transform(dft_shifted: np.ndarray) -> np.ndarray:
        """
        Reverse the Fourier transform and return the magnitude.
        
        Args:
            dft_shifted (np.ndarray): DFT shifted array
            
        Returns:
            np.ndarray: Reconstructed image
        """
        dft_shifted = np.fft.ifftshift(dft_shifted)  # Shift back
        img_back = cv2.idft(dft_shifted)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # Compute magnitude
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)  # Normalize for display
    
    @staticmethod
    def get_magnitude_spectrum(dft_shifted: np.ndarray) -> np.ndarray:
        """
        Calculate and normalize the magnitude spectrum for visualization.
        
        Args:
            dft_shifted (np.ndarray): DFT shifted array
            
        Returns:
            np.ndarray: Magnitude spectrum for display
        """
        magnitude = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return magnitude
    
    @staticmethod
    def create_filter(shape: Tuple[int, int], filter_type: str, 
                     cut_off_frequency: float, order: int = 2, 
                     highpass: bool = False, w: float = 10) -> np.ndarray:
        """
        Create various frequency filters in the frequency domain.
        
        Args:
            shape (tuple): Shape of the image (rows, cols)
            filter_type (str): Type of filter ('ideal', 'butterworth', 'gaussian', 
                              'lowpass', 'highpass', 'bandpass', 'bandstop')
            cut_off_frequency (float): Cutoff frequency
            order (int): Order for Butterworth filter
            highpass (bool): Whether to create highpass version
            w (float): Bandwidth for band filters
            
        Returns:
            np.ndarray: Filter mask
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.float32)
        
        for u in range(rows):
            for v in range(cols):
                d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)  # Distance from center
                
                if filter_type in ['ideal', 'lowpass']:
                    mask[u, v] = 1 if d <= cut_off_frequency else 0
                    
                elif filter_type == 'butterworth':
                    if d == 0:  # Avoid division by zero
                        mask[u, v] = 1
                    else:
                        mask[u, v] = 1 / (1 + (d / cut_off_frequency) ** (2 * order))
                        
                elif filter_type == 'gaussian':
                    mask[u, v] = np.exp(-(d**2) / (2 * (cut_off_frequency**2)))
                    
                elif filter_type == 'highpass':
                    mask[u, v] = 0 if d <= cut_off_frequency else 1
                    
                elif filter_type == 'bandpass':
                    mask[u, v] = 1 if (cut_off_frequency - w/2) <= d <= (cut_off_frequency + w/2) else 0
                    
                elif filter_type == 'bandstop':
                    mask[u, v] = 0 if (cut_off_frequency - w/2) <= d <= (cut_off_frequency + w/2) else 1
        
        # Convert to highpass if requested
        if highpass and filter_type in ['ideal', 'butterworth', 'gaussian']:
            mask = 1 - mask
            
        return mask
    
    def apply_frequency_filter(self, image: np.ndarray, filter_type: str, 
                              cut_off_frequency: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply frequency domain filter to an image.
        
        Args:
            image (np.ndarray): Input grayscale image
            filter_type (str): Type of filter to apply
            cut_off_frequency (float): Cutoff frequency
            **kwargs: Additional parameters for filter creation
            
        Returns:
            tuple: (filtered_image, filter_mask)
        """
        # Apply Fourier Transform
        dft_shifted = self.apply_fourier_transform(image)
        
        # Create filter
        filter_mask = self.create_filter(image.shape, filter_type, cut_off_frequency, **kwargs)
        
        # Apply filter (expand dimensions if needed)
        if len(filter_mask.shape) == 2:
            filter_mask = filter_mask[:, :, np.newaxis]
        filtered_dft = dft_shifted * filter_mask
        
        # Inverse transform
        filtered_image = self.inverse_fourier_transform(filtered_dft)
        
        return filtered_image, filter_mask.squeeze()
    
    def compare_filters(self, image: np.ndarray, filter_types: list, 
                       cut_off_frequency: float, **kwargs) -> dict:
        """
        Compare different filter types on the same image.
        
        Args:
            image (np.ndarray): Input grayscale image
            filter_types (list): List of filter types to compare
            cut_off_frequency (float): Cutoff frequency
            **kwargs: Additional parameters
            
        Returns:
            dict: Dictionary containing filtered images and masks
        """
        results = {}
        
        for filter_type in filter_types:
            filtered_img, filter_mask = self.apply_frequency_filter(
                image, filter_type, cut_off_frequency, **kwargs
            )
            results[filter_type] = {
                'filtered_image': filtered_img,
                'filter_mask': filter_mask
            }
        
        return results
    
    def plot_comparison(self, image: np.ndarray, results: dict, 
                       title_prefix: str = "Filter") -> None:
        """
        Plot comparison of different filters.
        
        Args:
            image (np.ndarray): Original image
            results (dict): Results from compare_filters
            title_prefix (str): Prefix for plot titles
        """
        num_filters = len(results)
        fig, axes = plt.subplots(1, num_filters + 1, figsize=(5 * (num_filters + 1), 5))
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot filtered images
        for i, (filter_type, data) in enumerate(results.items(), 1):
            axes[i].imshow(data['filtered_image'], cmap='gray')
            axes[i].set_title(f'{title_prefix} - {filter_type.capitalize()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the FourierImageProcessor class.
    """
    # Initialize processor
    processor = FourierImageProcessor()
    
    # Load image (replace with your image URL or path)
    image_url = "https://drive.google.com/uc?id=1PKCyYW9FWhsQUchwjSlwQYsmoSASdFSy"
    image = processor.load_image(image_url)
    
    if image is not None:
        # Display original image
        processor.show_image(image, "Original Image")
        
        # Apply Fourier Transform and show magnitude spectrum
        dft_shifted = processor.apply_fourier_transform(image)
        magnitude_spectrum = processor.get_magnitude_spectrum(dft_shifted)
        processor.show_image(magnitude_spectrum, "Fourier Transform Magnitude Spectrum")
        
        # Compare different filter types
        filter_types = ['ideal', 'butterworth', 'gaussian']
        cut_off_frequency = 50
        
        # Low-pass filters comparison
        lowpass_results = processor.compare_filters(
            image, filter_types, cut_off_frequency, highpass=False, order=2
        )
        processor.plot_comparison(image, lowpass_results, "Low-pass")
        
        # High-pass filters comparison
        highpass_results = processor.compare_filters(
            image, filter_types, cut_off_frequency, highpass=True, order=2
        )
        processor.plot_comparison(image, highpass_results, "High-pass")
        
    else:
        print("Failed to load image. Please check the URL or path.")


if __name__ == "__main__":
    main()
