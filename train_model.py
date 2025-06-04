import os
import cv2
import numpy as np
import numba
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import hsv_to_rgb

def get_all_image_paths(directory):
    """Get all image file paths from a directory"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_paths = []
    
    if not os.path.exists(directory):
        print(f"[WARNING] Directory does not exist: {directory}")
        return []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            image_paths.append(os.path.join(directory, filename))
    
    return image_paths

class ImageProcessor:
    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def median_filter(image, kernel_size=3):
        """Apply median filter to an image with reflection padding"""
        pad_size = kernel_size // 2
        h, w, c = image.shape
        output = np.zeros_like(image)
        
        # Manual padding with reflection
        padded_img = np.zeros((h + 2*pad_size, w + 2*pad_size, c), dtype=image.dtype)
        padded_img[pad_size:h+pad_size, pad_size:w+pad_size, :] = image
        
        # Reflect padding
        for i in range(pad_size):
            # Top and bottom borders
            padded_img[i, :, :] = padded_img[2*pad_size-i, :, :]
            padded_img[h+pad_size+i, :, :] = padded_img[h+pad_size-2-i, :, :]
        for j in range(pad_size):
            # Left and right borders
            padded_img[:, j, :] = padded_img[:, 2*pad_size-j, :]
            padded_img[:, w+pad_size+j, :] = padded_img[:, w+pad_size-2-j, :]
        
        # Pre-compute kernel area
        k_area = kernel_size * kernel_size
        mid_pos = k_area // 2
        
        # Process each channel in parallel
        for ch in numba.prange(c):
            for i in range(h):
                for j in range(w):
                    # Extract patch
                    patch = padded_img[i:i+kernel_size, j:j+kernel_size, ch]
                    
                    # Flatten and find median without full sort
                    flat_patch = patch.ravel()
                    
                    # Partial sort just to find median
                    if k_area % 2 == 1:
                        # For odd-sized kernels, just find the middle element
                        for k in range(mid_pos + 1):
                            min_idx = k
                            for m in range(k + 1, k_area):
                                if flat_patch[m] < flat_patch[min_idx]:
                                    min_idx = m
                            if min_idx != k:
                                flat_patch[k], flat_patch[min_idx] = flat_patch[min_idx], flat_patch[k]
                        output[i, j, ch] = flat_patch[mid_pos]
                    else:
                        # For even-sized kernels, find two middle elements
                        for k in range(mid_pos + 1):
                            min_idx = k
                            for m in range(k + 1, k_area):
                                if flat_patch[m] < flat_patch[min_idx]:
                                    min_idx = m
                            if min_idx != k:
                                flat_patch[k], flat_patch[min_idx] = flat_patch[min_idx], flat_patch[k]
                        output[i, j, ch] = (flat_patch[mid_pos-1] + flat_patch[mid_pos]) / 2
        
        return output

    @staticmethod
    def rgb_to_hsv(rgb_img):
        """Convert RGB image to HSV color space manually"""
        # Normalize RGB values to [0, 1] range
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

        # Value (V) is the maximum of R, G, B
        v = np.max(rgb_norm, axis=2)

        # Delta is the difference between max and min of R, G, B
        min_rgb = np.min(rgb_norm, axis=2)
        delta = v - min_rgb

        # Initialize Saturation (S) with zeros
        s = np.zeros_like(v)
        non_zero_v_mask = v != 0
        s[non_zero_v_mask] = delta[non_zero_v_mask] / v[non_zero_v_mask]

        # Initialize Hue (H) with zeros
        h = np.zeros_like(v)
        non_zero_delta_mask = delta != 0

        # Masks for determining dominant color
        red_max_mask = (v == r) & non_zero_delta_mask
        green_max_mask = (v == g) & non_zero_delta_mask
        blue_max_mask = (v == b) & non_zero_delta_mask

        # Calculate Hue based on dominant channel
        h[red_max_mask] = ((g[red_max_mask] - b[red_max_mask]) / delta[red_max_mask]) % 6
        h[green_max_mask] = 2 + (b[green_max_mask] - r[green_max_mask]) / delta[green_max_mask]
        h[blue_max_mask] = 4 + (r[blue_max_mask] - g[blue_max_mask]) / delta[blue_max_mask]

        # Convert H from degrees to [0, 1] range
        h = h * 60         # From 0-6 to 0-360 degrees
        h[h < 0] += 360    # Ensure positive values
        h = h / 360.0      # Scale to [0, 1] range

        return np.stack([h, s, v], axis=2)

    @staticmethod
    def preprocess_image(image_path):
        """Load and preprocess an image"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Could not read image: {image_path}")
            return None
        img = img[:, :, ::-1]  # Convert BGR to RGB
        img = ImageProcessor.median_filter(img, kernel_size=5)
        return img
    
    @staticmethod
    def preprocess_image_array(img_array):
        """Preprocess image from numpy array (for Streamlit uploaded files)"""
        if img_array is None:
            return None
        # img_array should already be in RGB format from PIL
        img_array = np.array(img_array)
        img_array = ImageProcessor.median_filter(img_array, kernel_size=5)
        return img_array

class FeatureExtractor:
    @staticmethod
    def extract_features(preprocessed_img):
        """Extract color features from preprocessed image"""
        hsv_img = ImageProcessor.rgb_to_hsv(preprocessed_img)
        hue_channel = hsv_img[:, :, 0]
        s_channel = hsv_img[:, :, 1]
        v_channel = hsv_img[:, :, 2]
        
        # Create mask for significant color areas
        mask = (s_channel > 0.3) & (v_channel > 0.3)
        masked_hue = hue_channel[mask]
        if len(masked_hue) == 0:
            masked_hue = hue_channel.flatten()

        return {
            'hue_mean': np.mean(hue_channel),
            'hue_std': np.std(hue_channel),
            'hue_median': np.median(hue_channel),
            'significant_hue_mean': np.mean(masked_hue),
            'significant_hue_std': np.std(masked_hue),
            'hsv_img': hsv_img
        }

class Classifier:
    def __init__(self):
        self.categories = []
        self.hue_ranges = {}

    def train(self, train_dir):
        """Train classifier on images in train_dir"""
        print("Training the classifier...")
        for category in self.categories:
            print(f"Processing category: {category}")
            category_features = []
            image_paths = get_all_image_paths(os.path.join(train_dir, category))
            for img_path in image_paths:
                preprocessed_img = ImageProcessor.preprocess_image(img_path)
                if preprocessed_img is not None:
                    features = FeatureExtractor.extract_features(preprocessed_img)
                    category_features.append(features)
            
            if len(category_features) > 0:
                hue_means = [f['significant_hue_mean'] for f in category_features]
                hue_stds = [f['significant_hue_std'] for f in category_features]
                avg_hue_mean = np.mean(hue_means)
                avg_hue_std = np.mean(hue_stds)
                
                self.hue_ranges[category] = {
                    'min': max(0, avg_hue_mean - 2 * avg_hue_std),
                    'max': min(1, avg_hue_mean + 2 * avg_hue_std),
                    'mean': avg_hue_mean,
                    'std': avg_hue_std
                }
                print(f"  {category} - Hue range: {self.hue_ranges[category]['min']:.3f} to {self.hue_ranges[category]['max']:.3f}")
            else:
                print(f"[WARNING] No training data for category: {category}")
        print("Training completed.")

    def classify_image_array(self, img_array):
        """Classify a single image from numpy array"""
        preprocessed_img = ImageProcessor.preprocess_image_array(img_array)
        if preprocessed_img is None:
            return None, None
            
        features = FeatureExtractor.extract_features(preprocessed_img)
        best_match = None
        min_distance = float('inf')
        
        for category, hue_range in self.hue_ranges.items():
            hue_mean = features['significant_hue_mean']
            hue_std = features['significant_hue_std']
            hue_data = features['hsv_img'][:, :, 0]
            s_channel = features['hsv_img'][:, :, 1]
            v_channel = features['hsv_img'][:, :, 2]
            mask = (s_channel > 0.3) & (v_channel > 0.3)
            masked_hue = hue_data[mask]

            hist, bins = np.histogram(masked_hue, bins=50, range=(0, 1))
            most_freq_hue_bin_idx = np.argmax(hist)
            bin_center = (bins[most_freq_hue_bin_idx] + bins[most_freq_hue_bin_idx + 1]) / 2

            hue_distance = abs(hue_mean - hue_range['mean'])
            std_threshold = 0.05
            std_penalty = 0.1 if (('rotten' in category and hue_std < std_threshold) or
                                  ('fresh' in category and hue_std > std_threshold)) else 0.0
            hue_penalty = 0.1 if not (hue_range['min'] <= bin_center <= hue_range['max']) else 0.0
            total_score = hue_distance + std_penalty + hue_penalty

            if total_score < min_distance:
                min_distance = total_score
                best_match = category

        return best_match, features

    def classify_image(self, image_path):
        """Classify a single image based on trained hue ranges"""
        preprocessed_img = ImageProcessor.preprocess_image(image_path)
        if preprocessed_img is None:
            return None, None
            
        features = FeatureExtractor.extract_features(preprocessed_img)
        best_match = None
        min_distance = float('inf')
        
        for category, hue_range in self.hue_ranges.items():
            hue_mean = features['significant_hue_mean']
            hue_std = features['significant_hue_std']
            hue_data = features['hsv_img'][:, :, 0]
            s_channel = features['hsv_img'][:, :, 1]
            v_channel = features['hsv_img'][:, :, 2]
            mask = (s_channel > 0.3) & (v_channel > 0.3)
            masked_hue = hue_data[mask]

            hist, bins = np.histogram(masked_hue, bins=50, range=(0, 1))
            most_freq_hue_bin_idx = np.argmax(hist)
            bin_center = (bins[most_freq_hue_bin_idx] + bins[most_freq_hue_bin_idx + 1]) / 2

            hue_distance = abs(hue_mean - hue_range['mean'])
            std_threshold = 0.05
            std_penalty = 0.1 if (('rotten' in category and hue_std < std_threshold) or
                                  ('fresh' in category and hue_std > std_threshold)) else 0.0
            hue_penalty = 0.1 if not (hue_range['min'] <= bin_center <= hue_range['max']) else 0.0
            total_score = hue_distance + std_penalty + hue_penalty

            if total_score < min_distance:
                min_distance = total_score
                best_match = category

        return best_match, features

def train_and_save_models():
    """Train models for all fruit types and save them"""
    base_dir = r"datasets"
    train_dir = os.path.join(base_dir, "train")
    
    fruit_types = ['apples', 'banana', 'oranges']
    models = {}
    
    for fruit in fruit_types:
        print(f"\n{'='*30}")
        print(f"==== Training {fruit.upper()} ====")
        print(f"{'='*30}\n")
        
        categories = [f"fresh{fruit}", f"rotten{fruit}"]
        
        classifier = Classifier()
        classifier.categories = categories
        classifier.train(train_dir)
        
        models[fruit] = classifier
        
        print(f"\n{fruit.capitalize()} training completed.")
    
    # Save all models
    with open('fruit_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("\n" + "="*30)
    print("=== ALL MODELS SAVED ===")
    print("="*30)
    print("Models saved to 'fruit_models.pkl'")

if __name__ == "__main__":
    train_and_save_models()