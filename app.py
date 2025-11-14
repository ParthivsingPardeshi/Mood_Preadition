"""
Emotion Detection Streamlit App with Beautiful UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data on first run
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return True

download_nltk_data()


# Add utils to path
sys.path.append('utils')
from preprocessing import TextPreprocessor

# Page configuration
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 16px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
        font-size: 18px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.05);
    }
    .emotion-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 15px 0;
        text-align: center;
    }
    h1 {
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model components
@st.cache_resource
def load_models():
    try:
        with open('models/emotion_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models/label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)
        return model, vectorizer, preprocessor, label_mapping
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run 'python train_model.py' first to train the model.")
        return None, None, None, None

model, vectorizer, preprocessor, EMOTION_LABELS = load_models()

# Emotion emojis and colors
EMOTION_CONFIG = {
    'sadness': {'emoji': '😢', 'color': '#4A90E2'},
    'joy': {'emoji': '😄', 'color': '#F5A623'},
    'love': {'emoji': '❤️', 'color': '#E91E63'},
    'anger': {'emoji': '😠', 'color': '#D0021B'},
    'fear': {'emoji': '😨', 'color': '#9013FE'},
    'surprise': {'emoji': '😮', 'color': '#50E3C2'}
}

# Title
st.title("🎭 Emotion Detection AI Dashboard")
st.markdown("### Analyze emotions in text with AI-powered NLP")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/667eea/ffffff?text=Emotion+AI", use_container_width=True)
    st.header("📊 Navigation")
    page = st.radio(
        "Choose a page",
        ["🔍 Real-time Detection", "📈 Emotion Trends", "📂 Batch Analysis", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    This app uses advanced NLP and Machine Learning to detect 6 emotions:
    - 😢 Sadness
    - 😄 Joy
    - ❤️ Love
    - 😠 Anger
    - 😨 Fear
    - 😮 Surprise
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ using Streamlit**")

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# Prediction function
def predict_emotion(text):
    """Predict emotion from text"""
    if not text.strip():
        return None, None
    
    clean_text = preprocessor.clean_text(text)
    if not clean_text.strip():
        return None, None
    
    text_vec = vectorizer.transform([clean_text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # Create emotions dictionary
    emotions_dict = {}
    for idx, emotion in enumerate(model.classes_):
        emotion_name = EMOTION_LABELS[emotion]
        emotions_dict[emotion_name] = probabilities[idx]
    
    predicted_emotion = EMOTION_LABELS[prediction]
    return predicted_emotion, emotions_dict

# PAGE 1: Real-time Detection
if "🔍 Real-time Detection" in page:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Enter Your Text")
        user_input = st.text_area(
            "Type or paste text here...",
            height=200,
            placeholder="Example: I am so happy today! This is amazing!",
            key="text_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            analyze_button = st.button("🔍 Analyze", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.text_input = ""
            st.rerun()
        
        if analyze_button and user_input and model:
            with st.spinner("Analyzing emotion..."):
                emotion, probs = predict_emotion(user_input)
                
                if emotion and probs:
                    # Save to history
                    st.session_state.emotion_history.append({
                        'text': user_input[:100] + '...' if len(user_input) > 100 else user_input,
                        'emotion': emotion,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'confidence': probs[emotion]
                    })
                    
                    # Display result
                    st.markdown("---")
                    st.markdown("### 🎯 Detection Result")
                    
                    # Main emotion card
                    emoji = EMOTION_CONFIG[emotion]['emoji']
                    color = EMOTION_CONFIG[emotion]['color']
                    
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h1 style="font-size: 80px; margin: 0;">{emoji}</h1>
                        <h2 style="color: {color}; margin: 10px 0;">{emotion.upper()}</h2>
                        <p style="font-size: 24px; color: #666;">
                            Confidence: {probs[emotion]*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability bar chart
                    st.markdown("### 📊 Emotion Probability Distribution")
                    
                    sorted_emotions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    emotions_list = [e[0] for e in sorted_emotions]
                    values_list = [e[1] for e in sorted_emotions]
                    colors_list = [EMOTION_CONFIG[e]['color'] for e in emotions_list]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=emotions_list,
                            y=values_list,
                            marker=dict(color=colors_list),
                            text=[f'{v*100:.1f}%' for v in values_list],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="Emotions",
                        yaxis_title="Probability",
                        height=400,
                        template="plotly_white",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Radar chart
                    st.markdown("### 🎯 Emotion Radar")
                    
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=list(probs.values()),
                        theta=list(probs.keys()),
                        fill='toself',
                        marker=dict(color='#667eea'),
                        line=dict(color='#764ba2', width=2)
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        height=450,
                        template="plotly_white",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Unable to analyze this text. Please try with different text.")
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.emotion_history:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝 Total Analyses</h3>
                <h2>{len(st.session_state.emotion_history)}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            emotions_list = [item['emotion'] for item in st.session_state.emotion_history]
            most_common = max(set(emotions_list), key=emotions_list.count)
            emoji = EMOTION_CONFIG[most_common]['emoji']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Most Common</h3>
                <h2>{emoji} {most_common.title()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if len(st.session_state.emotion_history) > 0:
                avg_conf = np.mean([item['confidence'] for item in st.session_state.emotion_history])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Avg Confidence</h3>
                    <h2>{avg_conf*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

# PAGE 2: Emotion Trends
elif "📈 Emotion Trends" in page:
    st.markdown("### 📈 Emotion Analysis Over Time")
    
    if st.session_state.emotion_history:
        df_history = pd.DataFrame(st.session_state.emotion_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            emotion_counts = df_history['emotion'].value_counts()
            colors = [EMOTION_CONFIG[e]['color'] for e in emotion_counts.index]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=emotion_counts.index,
                values=emotion_counts.values,
                marker=dict(colors=colors),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig_pie.update_layout(
                title="Emotion Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                df_history['emotion'].value_counts().reset_index(),
                x='emotion',
                y='count',
                color='emotion',
                color_discrete_map={e: EMOTION_CONFIG[e]['color'] for e in EMOTION_CONFIG},
                title="Emotion Frequency"
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline
        st.markdown("### ⏰ Emotion Timeline")
        
        fig_timeline = px.scatter(
            df_history,
            x='timestamp',
            y='emotion',
            color='emotion',
            size='confidence',
            hover_data=['text', 'confidence'],
            color_discrete_map={e: EMOTION_CONFIG[e]['color'] for e in EMOTION_CONFIG},
            title="Emotion Detection Timeline"
        )
        fig_timeline.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # History table
        st.markdown("### 📋 Analysis History")
        display_df = df_history[['timestamp', 'emotion', 'confidence', 'text']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"emotion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.emotion_history = []
            st.rerun()
    else:
        st.info("📝 No emotion data yet. Start analyzing text in the Real-time Detection page!")

# PAGE 3: Batch Analysis
elif "📂 Batch Analysis" in page:
    st.markdown("### 📂 Upload CSV for Batch Analysis")
    
    st.info("Upload a CSV file with a column containing text to analyze emotions in bulk.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        
        st.markdown("#### Preview of uploaded data:")
        st.dataframe(df_upload.head(10), use_container_width=True)
        
        text_column = st.selectbox("Select the text column to analyze", df_upload.columns)
        
        if st.button("🚀 Analyze All Texts"):
            with st.spinner(f"Analyzing {len(df_upload)} texts..."):
                emotions = []
                confidences = []
                
                progress_bar = st.progress(0)
                
                for idx, text in enumerate(df_upload[text_column]):
                    emotion, probs = predict_emotion(str(text))
                    if emotion and probs:
                        emotions.append(emotion)
                        confidences.append(probs[emotion])
                    else:
                        emotions.append('unknown')
                        confidences.append(0.0)
                    
                    progress_bar.progress((idx + 1) / len(df_upload))
                
                df_upload['detected_emotion'] = emotions
                df_upload['confidence'] = [f"{c*100:.1f}%" for c in confidences]
                
                st.success(f"✅ Analysis complete! Processed {len(df_upload)} texts.")
                
                # Display results
                st.markdown("#### Analysis Results:")
                st.dataframe(df_upload, use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df_upload,
                        x='detected_emotion',
                        color='detected_emotion',
                        color_discrete_map={e: EMOTION_CONFIG[e]['color'] for e in EMOTION_CONFIG},
                        title="Emotion Distribution in Dataset"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    emotion_counts = df_upload['detected_emotion'].value_counts()
                    colors = [EMOTION_CONFIG.get(e, {'color': '#gray'})['color'] for e in emotion_counts.index]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=emotion_counts.index,
                        values=emotion_counts.values,
                        marker=dict(colors=colors)
                    )])
                    fig_pie.update_layout(title="Emotion Distribution (%)")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Download results
                csv = df_upload.to_csv(index=False)
                st.download_button(
                    "📥 Download Analyzed Results",
                    csv,
                    f"emotion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

# PAGE 4: About
elif "ℹ️ About" in page:
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    ## 🎭 Emotion Detection AI
    
    This application uses **Natural Language Processing (NLP)** and **Machine Learning** 
    to detect emotions in text.
    
    ### 📊 Detected Emotions
    
    The model can detect **6 different emotions**:
    """)
    
    cols = st.columns(3)
    emotions_info = [
        ('sadness', '😢', 'Expressing sadness, disappointment, or sorrow'),
        ('joy', '😄', 'Expressing happiness, excitement, or delight'),
        ('love', '❤️', 'Expressing love, affection, or care'),
        ('anger', '😠', 'Expressing anger, frustration, or annoyance'),
        ('fear', '😨', 'Expressing fear, worry, or anxiety'),
        ('surprise', '😮', 'Expressing surprise, shock, or amazement')
    ]
    
    for idx, (emotion, emoji, description) in enumerate(emotions_info):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="emotion-card">
                <h1 style="font-size: 50px; margin: 10px 0;">{emoji}</h1>
                <h3 style="color: {EMOTION_CONFIG[emotion]['color']}; margin: 5px 0;">
                    {emotion.title()}
                </h3>
                <p style="font-size: 14px; color: #666; margin: 10px 0;">
                    {description}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔧 Technology Stack
    
    - **Streamlit**: Interactive web framework
    - **Scikit-learn**: Machine learning library
    - **NLTK**: Natural language processing
    - **Plotly**: Interactive visualizations
    - **TF-IDF Vectorization**: Text feature extraction
    - **Logistic Regression**: Classification model
    
    ### 📈 Model Performance
    
    The model achieves high accuracy through:
    - Advanced text preprocessing
    - TF-IDF feature extraction
    - Multi-class logistic regression
    - Training on thousands of labeled examples
    
    ### 🚀 How to Use
    
    1. **Real-time Detection**: Enter text to instantly detect emotions
    2. **Emotion Trends**: View your emotion analysis history
    3. **Batch Analysis**: Upload CSV files for bulk emotion detection
    
    ### 👨‍💻 Developer Info
    
    Built with Python and Streamlit for educational and research purposes.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: black; padding: 20px;">
    <p>🎭 Emotion Detection AI | Powered by Machine Learning & NLP</p>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
