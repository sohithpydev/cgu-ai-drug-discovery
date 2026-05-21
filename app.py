import streamlit as st
import pandas as pd
import subprocess
import os
import tempfile
import shutil
import sys
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration for paths
MODEL_PATHS = [
    os.path.join(BASE_DIR, "best_chunk0.pt"),
    os.path.join(BASE_DIR, "best_chunk1.pt"),
    os.path.join(BASE_DIR, "best_chunk2.pt"),
]

# Set page configuration
st.set_page_config(
    page_title="AI Drug Discovery - CGU",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        color: #2E5090;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #eaebf0;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #2E5090;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .coming-soon {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: 600;
        margin-top: 20px;
    }
    .metric-callout {
        background-color: #e8f4f8;
        border: 1px solid #b6e0ed;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='main-title'>Chang Gung University<br>Office of AI Drug Discovery</h1>", unsafe_allow_html=True)

st.info("📢 **Announcement:** We are currently screening for **1 billion compounds**, and their respective graphs will be live soon!")

tab1, tab2 = st.tabs(["📊 Model Performance", "🚀 Click here to start screening your site"])

with tab2:
    # Target Site Selection
    st.subheader("🎯 Select Target Site")
    target_sites = ["APC-8B CBOX1", "APC-8B CBOX2", "APC8B KILR", "APC1", "APC5"]
    selected_site = st.selectbox("Choose a target to begin screening:", target_sites)
    
    if selected_site != "APC-8B CBOX1":
        st.markdown(f"<div class='coming-soon'>🚀 The model for <b>{selected_site}</b> is currently under development. Please check back later!</div>", unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.subheader("📂 Upload Screening Data")
        
        st.markdown("""
        <div class='info-box'>
            <b>Input Format:</b><br>
            Please provide a CSV file containing your <code>SMILES</code> strings and a unique <code>Compound_ID</code>.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Your `.csv` file must contain the following columns:")
        required_columns = ["Compound_ID", "SMILES"]
        display_columns = ["Compound_ID", "SMILES"]
        st.code(", ".join(display_columns), language="text")
        
        # Provide sample CSV for download
        sample_csv_path = os.path.join(BASE_DIR, "filtered_smiles_all_01_clean_3d_random100.csv")
        if os.path.exists(sample_csv_path):
            with open(sample_csv_path, "rb") as file:
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=file,
                    file_name="sample_input.csv",
                    mime="text/csv"
                )
        
        
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the CSV to validate columns
                df = pd.read_csv(uploaded_file)
                missing_cols = [col for col in required_columns if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Uploaded file is missing some required columns: {', '.join(missing_cols)}")
                else:
                    st.success(f"✅ Successfully loaded `{uploaded_file.name}` with {len(df)} compounds.")
                    
                    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                        with st.spinner("Preparing to run Chemprop prediction..."):
                            # Determine which models to use
                            models_to_use = MODEL_PATHS
                            if not all(os.path.exists(p) for p in models_to_use):
                                st.error("❌ Could not find the new model files (`best_chunk0.pt`, etc.). Please ensure they exist.")
                                st.stop()
                                    
                            # Save the uploaded file to a temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_input:
                                df.to_csv(tmp_input.name, index=False)
                                input_path = tmp_input.name
                                
                            # Set output path
                            output_path = os.path.abspath("ensemble_predictions.csv")
                            
                            chemprop_cmd = ["chemprop"]
                            if shutil.which("chemprop") is None:
                                possible_path = os.path.join(os.path.dirname(sys.executable), "chemprop")
                                if os.path.exists(possible_path):
                                    chemprop_cmd = [possible_path]
                                else:
                                    chemprop_cmd = ["conda", "run", "-n", "chemprop", "chemprop"]
                                    
                            # Build the command
                            command = chemprop_cmd + [
                                "predict",
                                "--test-path", input_path,
                                "--smiles-columns", "SMILES",
                                "--model-path"
                            ] + models_to_use + [
                                "--drop-extra-columns",
                                "--uncertainty-method", "ensemble",
                                "--accelerator", "cpu", # Safe default for local Mac testing
                                "--devices", "1",
                                "--num-workers", "0",
                                "--output", output_path
                            ]
                            
                        with st.spinner("Running Chemprop (this may take a moment)..."):
                            st.info("Executing command:\n\n`" + " ".join(command) + "`")
                            try:
                                # Execute the command
                                result = subprocess.run(command, capture_output=True, text=True)
                                
                                if result.returncode != 0:
                                    st.error("❌ Prediction failed.")
                                    st.code(result.stderr, language="bash")
                                else:
                                    st.success("✅ Prediction completed successfully!")
                                    
                                    # Show results
                                    if os.path.exists(output_path):
                                        results_df = pd.read_csv(output_path)
                                        st.subheader("📊 Prediction Results")
                                        st.dataframe(results_df, use_container_width=True)
                                        
                                        # Provide download button
                                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="📥 Download Full Results",
                                            data=csv_data,
                                            file_name="ensemble_predictions.csv",
                                            mime="text/csv",
                                            type="primary"
                                        )
                            except Exception as e:
                                st.error(f"❌ An error occurred while executing the command: {str(e)}")
                            finally:
                                # Cleanup temporary input file
                                if os.path.exists(input_path):
                                    os.remove(input_path)
                                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

with tab1:
    st.header("Ensemble Model Performance")
    
    st.markdown("""
    <div class='metric-callout'>
        The ensemble model was trained on a comprehensive dataset of <b>8.9 million</b> docking data points. 
        This data was divided into 3 equal chunks (~2.96 in each) to train 3 separate models, ensuring robust generalization and high predictive accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Individual Chunk Performance")
    st.markdown("All three models achieved an average Spearman rank of **0.94** on their respective test chunks.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Chunk 1")
        if os.path.exists("learning_curve_chunk_0.png"):
            st.image("learning_curve_chunk_0.png", caption="Learning Curve", use_container_width=True)
        if os.path.exists("actual_vs_predicted_chunk_0_scatter_plot.png"):
            st.image("actual_vs_predicted_chunk_0_scatter_plot.png", caption="Actual vs Predicted", use_container_width=True)
            
    with col2:
        st.subheader("Chunk 2")
        if os.path.exists("learning_curve_chunk_1.png"):
            st.image("learning_curve_chunk_1.png", caption="Learning Curve", use_container_width=True)
        if os.path.exists("actual_vs_predicted_chunk_1_scatter_plot.png"):
            st.image("actual_vs_predicted_chunk_1_scatter_plot.png", caption="Actual vs Predicted", use_container_width=True)
            
    with col3:
        st.subheader("Chunk 3")
        if os.path.exists("learning_curve_chunk_2.png"):
            st.image("learning_curve_chunk_2.png", caption="Learning Curve", use_container_width=True)
        if os.path.exists("actual_vs_predicted_chunk_2_scatter_plot.png"):
            st.image("actual_vs_predicted_chunk_2_scatter_plot.png", caption="Actual vs Predicted", use_container_width=True)
            
    st.markdown("---")
    
    st.markdown("### 🔄 Cross-Chunk Generalization")
    st.markdown("""
    To rigorously verify the models' ability to generalize to unseen data, **cross-chunk testing** was performed. 
    We tested the Chunk 2 and Chunk 3 models entirely on the Chunk 1 test set. 
    
    The cross-chunk tests also yielded an impressive average Spearman rank of **0.94**, demonstrating excellent model generalization.
    """)
    
    cross_col1, cross_col2 = st.columns(2)
    with cross_col1:
        st.subheader("Chunk 2 Model on Chunk 1 Test Set")
        if os.path.exists("actual_vs_predicted_cross_B_on_A_scatter_plot.png"):
            st.image("actual_vs_predicted_cross_B_on_A_scatter_plot.png", caption="Actual vs Predicted", use_container_width=True)
            
    with cross_col2:
        st.subheader("Chunk 3 Model on Chunk 1 Test Set")
        if os.path.exists("actual_vs_predicted_cross_C_on_A_scatter_plot.png"):
            st.image("actual_vs_predicted_cross_C_on_A_scatter_plot.png", caption="Actual vs Predicted", use_container_width=True)
            
    st.markdown("---")
