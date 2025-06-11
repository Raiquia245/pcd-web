import streamlit as st
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import io

# Import the classes from train_model (make sure train_model.py is in the same directory)
from train_model import ImageProcessor, FeatureExtractor, Classifier

# Configure the page
st.set_page_config(
    page_title="Fruit Ripeness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fruit-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #E8E8E8;
        margin: 1rem 0;
    }
    .fresh-result {
        background-color: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28A745;
    }
    .rotten-result {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #DC3545;
    }
    .feature-info {
        background-color: #F8F9FA;
        color: #212529;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        with open('fruit_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'train_model.py' first to train the models.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

def main():
    # Header
    st.markdown('<h1 class="main-header">🍎 Fruit Ripeness Classifier 🍌</h1>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    # Sidebar for fruit selection
    st.sidebar.header("🔧 Configuration")
    
    fruit_options = {
        'Apples 🍎': 'apples',
        'Bananas 🍌': 'banana', 
        'Oranges 🍊': 'oranges'
    }
    
    selected_fruit_display = st.sidebar.selectbox(
        "Select fruit type:",
        list(fruit_options.keys())
    )
    selected_fruit = fruit_options[selected_fruit_display]
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header(f"📤 Upload {selected_fruit_display}")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of the fruit for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add classification button
            if st.button("🔍 Classify Fruit", type="primary", use_container_width=True):
                classify_fruit(image, selected_fruit, models[selected_fruit], col2)
                st.session_state['result_shown'] = True
    
    with col2:
        if 'result_shown' not in st.session_state:
            st.header("📊 Classification Results")
            st.info("Upload an image and click 'Classify Fruit' to see results here.")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About This Classifier")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        **🎯 How it works:**
        - Extracts color features (HSV)
        - Analyzes hue distribution
        - Compares with trained patterns
        - Classifies as fresh or rotten
        """)
    
    with info_col2:
        st.markdown("""
        **🍎 Supported Fruits:**
        - Apples
        - Bananas  
        - Oranges
        """)
    
    with info_col3:
        st.markdown("""
        **📝 Tips for best results:**
        - Use clear, well-lit images
        - Ensure fruit fills most of frame
        - Avoid blurry or dark images
        """)
def calculate_accurate_confidence(features, model):
    """
    Menghitung confidence menggunakan algoritma yang sama seperti model classification
    """
    confidence_scores = {}
    
    # Ekstrak data yang diperlukan
    hue_data = features['hsv_img'][:, :, 0]
    s_channel = features['hsv_img'][:, :, 1]
    v_channel = features['hsv_img'][:, :, 2]
    mask = (s_channel > 0.3) & (v_channel > 0.3)
    masked_hue = hue_data[mask]
    
    # Hitung histogram untuk mendapatkan most frequent hue
    hist, bins = np.histogram(masked_hue, bins=50, range=(0, 1))
    most_freq_hue_bin_idx = np.argmax(hist)
    bin_center = (bins[most_freq_hue_bin_idx] + bins[most_freq_hue_bin_idx + 1]) / 2
    
    hue_mean = features['significant_hue_mean'] 
    hue_std = features['significant_hue_std']
    
    # Hitung score untuk setiap kategori menggunakan ALGORITMA YANG SAMA
    scores = {}
    for category, hue_range in model.hue_ranges.items():
        # 1. Hue distance
        hue_distance = abs(hue_mean - hue_range['mean'])
        
        # 2. Standard deviation penalty
        std_threshold = 0.05
        std_penalty = 0.1 if (('rotten' in category and hue_std < std_threshold) or
                              ('fresh' in category and hue_std > std_threshold)) else 0.0
        
        # 3. Hue penalty (berdasarkan histogram)
        hue_penalty = 0.1 if not (hue_range['min'] <= bin_center <= hue_range['max']) else 0.0
        
        # 4. Total score (semakin kecil = semakin cocok)
        total_score = hue_distance + std_penalty + hue_penalty
        scores[category] = total_score
    
    # Konversi score ke confidence (inverse relationship)
    # Score rendah = confidence tinggi
    max_score = max(scores.values()) + 0.001  # Avoid division by zero
    
    for category, score in scores.items():
        # Confidence = 1 - (normalized_score)
        normalized_score = score / max_score
        confidence_scores[category] = (1 - normalized_score) * 100
    
    # Normalize agar total = 100%
    total_confidence = sum(confidence_scores.values())
    if total_confidence > 0:
        for category in confidence_scores:
            confidence_scores[category] = confidence_scores[category] / total_confidence * 100
    
    return confidence_scores, scores

def classify_fruit(image, fruit_type, model, result_column):
    with result_column:
        st.header("📊 Classification Results")  
        with st.spinner("🔄 Analyzing image..."):
            try:
                # Convert PIL image to numpy array
                img_array = np.array(image)
                
                # Classify the image
                predicted_class, features = model.classify_image_array(img_array)
                
                if predicted_class is None:
                    st.error("❌ Could not classify the image. Please try another image.")
                    return
                
                # Determine if fresh or rotten
                is_fresh = "fresh" in predicted_class.lower()
                fruit_status = "Fresh" if is_fresh else "Rotten"
                
                # Display result with appropriate styling
                if is_fresh:
                    st.markdown(f"""
                    <div class="fresh-result">
                        <h2>✅ Result: {fruit_status}</h2>
                        <p>This {fruit_type} appears to be <strong>fresh</strong> and good to eat!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rotten-result">
                        <h2>⚠️ Result: {fruit_status}</h2>
                        <p>This {fruit_type} appears to be <strong>rotten</strong>. Consider discarding it.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display feature information
                st.markdown("### 📈 Analysis Details")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"""
                    <div class="feature-info">
                        <h4>Color Features:</h4>
                        <ul>
                            <li><strong>Hue Mean:</strong> {features['hue_mean']:.3f}</li>
                            <li><strong>Hue Std:</strong> {features['hue_std']:.3f}</li>
                            <li><strong>Significant Hue Mean:</strong> {features['significant_hue_mean']:.3f}</li>
                            <li><strong>Significant Hue Std:</strong> {features['significant_hue_std']:.3f}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="feature-info">
                        <h4>Classification Info:</h4>
                        <ul>
                            <li><strong>Predicted Class:</strong> {predicted_class}</li>
                            <li><strong>Fruit Type:</strong> {fruit_type.capitalize()}</li>
                            <li><strong>Status:</strong> {fruit_status}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualize HSV components
                st.markdown("### 🎨 Color Analysis Visualization")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img_array)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Hue channel
                hue_img = features['hsv_img'][:, :, 0]
                axes[1].imshow(hue_img, cmap='hsv')
                axes[1].set_title(f'Hue Channel\nMean: {features["significant_hue_mean"]:.3f}')
                axes[1].axis('off')
                
                # HSV image
                hsv_rgb = hsv_to_rgb(features['hsv_img'])
                axes[2].imshow(hsv_rgb)
                axes[2].set_title('HSV Representation')
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # # CONFIDENCE CALCULATION YANG AKURAT
                # st.markdown("### 📊 Accurate Model Confidence")
                
                # confidence_scores, raw_scores = calculate_accurate_confidence(features, model)
                
                # categories = list(confidence_scores.keys())
                # confidences = list(confidence_scores.values())
                
                # # Display confidence bars
                # fig, ax = plt.subplots(figsize=(10, 4))
                # colors = ['green' if 'fresh' in cat.lower() else 'red' for cat in categories]
                # bars = ax.bar(categories, confidences, color=colors, alpha=0.7)
                
                # # Highlight predicted class
                # predicted_idx = None
                # for i, cat in enumerate(categories):
                #     if cat.lower() == predicted_class.lower():
                #         bars[i].set_edgecolor('black')
                #         bars[i].set_linewidth(3)
                #         predicted_idx = i
                #         break
                
                # ax.set_ylabel('Confidence (%)')
                # ax.set_title('Classification Confidence (Using Actual Model Algorithm)')
                # ax.set_ylim(0, 100)
                
                # # Add value labels on bars
                # for i, (bar, conf) in enumerate(zip(bars, confidences)):
                #     label = f'{conf:.1f}%' + (' ✓' if i == predicted_idx else '')
                #     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                #         label, ha='center', va='center', color='white' if conf > 30 else 'black',
                #         weight='bold' if i == predicted_idx else 'normal', fontsize=10)
                
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # st.pyplot(fig)

                

                
                
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
                st.error("Please make sure the image is valid and try again.")

if __name__ == "__main__":
    main()