import streamlit as st
import joblib
import numpy as np
import time
import psutil
import platform
import os
import csv
import statistics
from datetime import datetime
import pandas as pd
from fpdf import FPDF  # pip install fpdf for PDF generation

# Page config for professional look
st.set_page_config(
    page_title="NefroAi: CKD Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-indicator {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .high-risk { background-color: #ffebee; color: #c62828; }
    .low-risk { background-color: #e8f5e8; color: #2e7d32; }
    .normal { background-color: #fff3e0; color: #ef6c00; }
    .stDataFrame > div > div > div > table {
        border-collapse: collapse;
        border: 1px solid #ddd;
    }
    .stDataFrame > div > div > div > table th {
        background-color: #f0f2f6;
        text-align: center;
        font-weight: bold;
    }
    .stDataFrame > div > div > div > table td {
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Default input data
default_input_data = [10.1, 30, 4.5, 1.2, 1.015, 110, 2.5, 135, 4.5, 45.0]


# =========================
# Utility Functions
# =========================
@st.cache_data
def now_us():
    return time.perf_counter() * 1_000_000  # microseconds


@st.cache_data
def system_snapshot():
    v = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": int((v.total - v.available) / (1024 * 1024)),
        "ram_total_mb": int(v.total / (1024 * 1024)),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def log_latency(times, extra):
    os.makedirs("logs", exist_ok=True)
    path = "logs/latency_log.csv"
    file_exists = os.path.isfile(path)
    fieldnames = list({**times, **extra}.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**times, **extra})


# =========================
# Model Management
# =========================
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Global model variable
if 'model' not in st.session_state:
    st.session_state.model = load_model("best_rf_model.pkl")


# =========================
# Prediction Function with Timing
# =========================
def predict(input_data, model):
    if model is None:
        return None, None, None
    t0 = now_us()
    # Preprocess: converting list to np.array
    X = np.array([input_data])
    t1 = now_us()
    prediction = model.predict(X)
    t2 = now_us()
    probability = model.predict_proba(X)
    t3 = now_us()

    # Convert to ms with higher precision
    times = {
        "preprocess_ms": round((t1 - t0) / 1000, 3),
        "inference_ms": round((t2 - t1) / 1000, 3),
        "probability_ms": round((t3 - t2) / 1000, 3),
        "total_ms": round((t3 - t0) / 1000, 3)
    }
    return prediction[0], probability[0][1], times


# Risk Table Data based on thresholds (comprehensive for all features)
def get_risk_table_data(input_data):
    hemo, pcv, rc, sc, sg, bgr, al, sod, pot, bu = input_data

    features_data = [
        {
            "Feature": "Hemoglobin",
            "Value": hemo,
            "Unit": "g/dL",
            "Normal Range": "12-16",
            "Risk": "Low" if 12 <= hemo <= 16 else "High",
            "Role": "Indicator of anemia often associated with CKD",
            "Interpretation": "Low levels suggest anemia due to kidney dysfunction; monitor for fatigue and pallor."
        },
        {
            "Feature": "Packed Cell Volume (PCV)",
            "Value": pcv,
            "Unit": "%",
            "Normal Range": "36-46",
            "Risk": "Low" if 36 <= pcv <= 46 else "High",
            "Role": "Measures blood volume occupied by red blood cells",
            "Interpretation": "Abnormal levels can indicate dehydration or anemia linked to CKD progression."
        },
        {
            "Feature": "Red Blood Cell Count (RBC)",
            "Value": rc,
            "Unit": "10^6 cells/uL",
            "Normal Range": "4.5-5.9",
            "Risk": "Low" if 4.5 <= rc <= 5.9 else "High",
            "Role": "Assesses oxygen-carrying capacity",
            "Interpretation": "Low counts contribute to anemia in advanced CKD; supplementation may be needed."
        },
        {
            "Feature": "Serum Creatinine",
            "Value": sc,
            "Unit": "mg/dL",
            "Normal Range": "0.6-1.2",
            "Risk": "Low" if 0.6 <= sc <= 1.2 else "High",
            "Role": "Key marker of kidney filtration function",
            "Interpretation": "Elevated levels signal impaired glomerular filtration rate (GFR); urgent evaluation required."
        },
        {
            "Feature": "Specific Gravity",
            "Value": sg,
            "Unit": "",
            "Normal Range": "1.010-1.020",
            "Risk": "Low" if 1.010 <= sg <= 1.020 else "High",
            "Role": "Indicates kidney's ability to concentrate urine",
            "Interpretation": "Deviations may point to tubular dysfunction or diabetes insipidus in CKD context."
        },
        {
            "Feature": "Blood Glucose Random",
            "Value": bgr,
            "Unit": "mg/dL",
            "Normal Range": "<140",
            "Risk": "Low" if bgr < 140 else "High",
            "Role": "Screens for diabetes, a major CKD risk factor",
            "Interpretation": "High values increase risk of diabetic nephropathy; lifestyle modifications advised."
        },
        {
            "Feature": "Albumin",
            "Value": al,
            "Unit": "g/dL",
            "Normal Range": "3.5-5.0",
            "Risk": "Low" if 3.5 <= al <= 5.0 else "High",
            "Role": "Assesses nutritional status and liver/kidney function",
            "Interpretation": "Low levels indicate proteinuria or malnutrition, common in CKD stages."
        },
        {
            "Feature": "Sodium",
            "Value": sod,
            "Unit": "mEq/L",
            "Normal Range": "135-145",
            "Risk": "Low" if 135 <= sod <= 145 else "High",
            "Role": "Maintains fluid balance and nerve function",
            "Interpretation": "Imbalances in CKD due to dietary restrictions or medications; affects blood pressure."
        },
        {
            "Feature": "Potassium",
            "Value": pot,
            "Unit": "mEq/L",
            "Normal Range": "3.5-5.0",
            "Risk": "Low" if 3.5 <= pot <= 5.0 else "High",
            "Role": "Essential for heart and muscle function",
            "Interpretation": "Hyperkalemia common in CKD; monitor to prevent arrhythmias."
        },
        {
            "Feature": "Blood Urea",
            "Value": bu,
            "Unit": "mg/dL",
            "Normal Range": "7-20",
            "Risk": "Low" if 7 <= bu <= 20 else "High",
            "Role": "Byproduct of protein metabolism cleared by kidneys",
            "Interpretation": "Elevated in reduced kidney function; correlates with uremic symptoms."
        }
    ]

    return features_data


# =========================
# PDF Report Generation - Professional Version (without Risk Indicators)
# =========================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(31, 119, 180)  # Blue color
        self.cell(0, 10, 'NefroAi CKD Risk Assessment Report', 0, 1, 'C')
        self.ln(5)

        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'NefroAi | AI-Powered CKD Prediction Tool | For Informational Purposes Only', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(31, 119, 180)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_table(self, headers, data, col_widths, alignments=None):
        if alignments is None:
            alignments = ['C'] * len(headers)

        self.set_font('Arial', 'B', 10)
        self.set_fill_color(240, 242, 246)  # Light gray
        self.set_text_color(31, 119, 180)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln()

        self.set_font('Arial', '', 9)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        for row in data:
            for i, item in enumerate(row):
                align = alignments[i] if i < len(alignments) else 'L'
                self.cell(col_widths[i], 6, str(item), 1, 0, align)
            self.ln()


def generate_pdf_report(prediction, probability, times, snap, input_data):
    pdf = PDF()
    pdf.add_page()

    # Prediction Summary Section
    pdf.chapter_title('Prediction Summary')
    risk_status = "High Risk: Chronic Kidney Disease Detected" if prediction == 1 else "Low Risk: No CKD Detected"
    confidence = f"{(probability * 100 if prediction == 1 else (1 - probability) * 100):.1f}%"

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, f'Status: {risk_status}', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, f'Confidence Level: {confidence}', 0, 1)
    pdf.ln(5)

    # Input Parameters Table
    pdf.chapter_title('Clinical Parameters')
    params = ['Hemoglobin (g/dL)', 'PCV (%)', 'RBC Count (10^6/uL)', 'Serum Creatinine (mg/dL)', 'Specific Gravity',
              'Blood Glucose Random (mg/dL)', 'Albumin (g/dL)', 'Sodium (mEq/L)', 'Potassium (mEq/L)',
              'Blood Urea (mg/dL)']
    data = [[p, f"{v:.2f}"] for p, v in zip(params, input_data)]
    headers = ['Parameter', 'Value']
    col_widths = [100, 90]
    pdf.add_table(headers, data, col_widths)
    pdf.ln(5)

    # System Performance Section
    pdf.chapter_title('System Performance')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f'Total Inference Time: {times["total_ms"]} ms', 0, 1)
    pdf.cell(0, 6, f'CPU Usage: {snap["cpu_percent"]:.1f}%', 0, 1)
    pdf.cell(0, 6, f'RAM Usage: {snap["ram_used_mb"]} / {snap["ram_total_mb"]} MB', 0, 1)
    pdf.ln(5)

    # Disclaimer
    pdf.chapter_title('Important Disclaimer')
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(192, 0, 0)
    disclaimer = 'This report is generated using an AI model for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personalized medical guidance. The predictions are based on provided data and model estimates; actual health status may vary.'
    pdf.multi_cell(0, 4, disclaimer)

    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output


# =========================
# Batch Benchmark Function
# =========================
def batch_benchmark(samples, model, n_runs=100):
    if model is None:
        return None
    results = []
    for i in range(n_runs):
        _, _, times = predict(samples[i % len(samples)], model)
        if times:
            results.append(times["total_ms"])

    if not results:
        return None

    p50 = round(statistics.median(results), 3)
    p95 = round(np.percentile(results, 95), 3)
    mean = round(statistics.mean(results), 3)
    total_time_ms = sum(results)
    if total_time_ms > 0:
        throughput = round(n_runs / (total_time_ms / 1000), 2)  # req/sec
    else:
        throughput = float('inf')  # Indicate extremely fast

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean,
        "throughput_rps": throughput
    }


# =========================
# Sidebar: App Info and Model Management
# =========================
with st.sidebar:
    st.header("🩸 NefroAi Info")
    st.markdown("""
    **NefroAi** is an AI-powered tool for predicting Chronic Kidney Disease (CKD) risk using key clinical biomarkers.  
    Built with Streamlit and scikit-learn.  
    For research/educational use only – consult a doctor for medical advice.
    """)

    st.header("🔧 Model Management")
    uploaded_file = st.file_uploader("Upload a new .pkl model", type="pkl")
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.model = load_model("temp_model.pkl")
            st.success("Model uploaded and loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

    if st.button("Reload Default Model"):
        st.session_state.model = load_model("best_rf_model.pkl")
        st.success("Default model reloaded!")

# =========================
# Main Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Clinical & Interpretive", "⚙️ Batch Benchmark"])

with tab1:
    st.markdown('<h1 class="main-header">NefroAi: CKD Risk Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
    Enter your clinical parameters below to assess CKD risk. Values are based on standard medical ranges.
    """)

    # Input form in columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Blood Parameters")
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.1, max_value=17.8, step=0.1, value=10.1,
                               help="Normal: 12-16 g/dL")
        pcv = st.number_input("Packed Cell Volume (%)", min_value=9, max_value=54, step=1, value=30,
                              help="Normal: 36-46%")
        rc = st.number_input("Red Blood Cell Count (10^6 cells/uL)", min_value=2.0, max_value=8.0, step=0.1, value=4.5,
                             help="Normal: 4.5-5.9")
        sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.4, max_value=76.1, step=0.1, value=1.2,
                             help="Normal: 0.6-1.2")
        sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.025, step=0.001, value=1.015,
                             help="Normal: 1.010-1.020")

    with col2:
        st.subheader("Metabolic Parameters")
        bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=22, max_value=490, step=1, value=110,
                              help="Normal: <140")
        al = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=5.0, step=0.1, value=2.5,
                             help="Normal: 3.5-5.0")
        sod = st.number_input("Sodium (mEq/L)", min_value=104, max_value=163, step=1, value=135, help="Normal: 135-145")
        pot = st.number_input("Potassium (mEq/L)", min_value=2.5, max_value=7.0, step=0.1, value=4.5,
                              help="Normal: 3.5-5.0")
        bu = st.number_input("Blood Urea (mg/dL)", min_value=1.5, max_value=391.1, step=1.0, value=45.0,
                             help="Normal: 7-20")

    input_data = [hemo, pcv, rc, sc, sg, bgr, al, sod, pot, bu]
    st.session_state.input_data = input_data

    # Predict button
    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        if st.button("🔍 Predict CKD Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing your data..."):
                prediction, probability, times = predict(input_data, st.session_state.model)
                if prediction is None:
                    st.error("Model not loaded. Please upload or reload a model.")
                else:
                    snap = system_snapshot()
                    log_latency(times, snap)

                    # Store for PDF
                    if 'report_data' not in st.session_state:
                        st.session_state.report_data = {}
                    risk_data = get_risk_table_data(input_data)
                    st.session_state.report_data = {
                        'prediction': prediction,
                        'probability': probability,
                        'times': times,
                        'snap': snap,
                        'input_data': input_data
                    }

                    # Prediction Display
                    col_pred, col_conf = st.columns([1, 2])
                    with col_pred:
                        if prediction == 1:
                            st.error("⚠️ **High Risk:** Chronic Kidney Disease detected.")
                        else:
                            st.success("✅ **Low Risk:** No CKD detected.")

                    with col_conf:
                        risk_pct = probability * 100 if prediction == 1 else (1 - probability) * 100
                        st.metric("Model Confidence", f"{risk_pct:.1f}%")

                    # Risk Indicators Table
                    st.subheader("📈 Risk Indicators")
                    df_risk = pd.DataFrame(risk_data)
                    display_df = df_risk[['Feature', 'Value', 'Risk', 'Role', 'Interpretation']].copy()


                    def color_risk(val):
                        if val == 'High':
                            return 'background-color: #ffebee; color: #c62828'
                        elif val == 'Low':
                            return 'background-color: #e8f5e8; color: #2e7d32'
                        else:
                            return ''


                    styled_df = display_df.style.set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f2f6')]},
                        {'selector': 'td', 'props': [('text-align', 'left')]},
                    ]).set_properties(**{
                        'Feature': {'text-align': 'left', 'width': '200px'},
                        'Value': {'text-align': 'center', 'width': '80px', 'font-weight': 'bold'},
                        'Risk': {'text-align': 'center', 'width': '60px', 'font-weight': 'bold'},
                        'Role': {'text-align': 'left', 'width': '150px'},
                        'Interpretation': {'text-align': 'left', 'width': '300px'}
                    }).applymap(color_risk, subset=['Risk']).format({
                        'Value': '{:.2f}'
                    })
                    st.dataframe(styled_df, use_container_width=True)

                    # Performance Metrics
                    st.subheader("⚙️ Performance & System Snapshot")
                    col_lat, col_sys = st.columns(2)

                    with col_lat:
                        st.markdown('<div class="metric-card">**⏱️ Latency Breakdown (ms)**</div>',
                                    unsafe_allow_html=True)
                        st.metric("Preprocessing", times["preprocess_ms"])
                        st.metric("Inference", times["inference_ms"])
                        st.metric("Probability Calc.", times["probability_ms"])
                        st.metric("Total", times["total_ms"])

                    with col_sys:
                        st.markdown('<div class="metric-card">**💻 System Resources**</div>', unsafe_allow_html=True)
                        st.metric("CPU Usage", f"{snap['cpu_percent']:.1f}%")
                        st.metric("RAM Used", f"{snap['ram_used_mb']} / {snap['ram_total_mb']} MB")
                        with st.expander("Hardware Details"):
                            st.write(f"**OS:** {snap['os']}")
                            st.write(f"**Processor:** {snap['processor']}")
                            st.write(f"**Machine:** {snap['machine']}")

                    # Download PDF Report Button
                    st.subheader("📄 Report Download")
                    if 'report_data' in st.session_state:
                        pdf_data = generate_pdf_report(
                            st.session_state.report_data['prediction'],
                            st.session_state.report_data['probability'],
                            st.session_state.report_data['times'],
                            st.session_state.report_data['snap'],
                            st.session_state.report_data['input_data']
                        )
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name=f"nefroai_ckd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    with col_clear:
        if st.button("🗑️ Clear Inputs"):
            if 'report_data' in st.session_state:
                del st.session_state.report_data
            st.rerun()

with tab2:
    st.markdown('<h1 class="main-header">Clinical & Interpretive Insights</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### Understanding CKD Risk
    Chronic Kidney Disease (CKD) is a progressive condition where kidneys gradually lose function. Early detection via biomarkers like creatinine and urea is crucial.

    **Key Biomarkers Explained:**
    - **Serum Creatinine:** Waste product; high levels indicate poor kidney filtration.
    - **Blood Urea Nitrogen (BUN):** Measures urea; elevated in dehydration or kidney issues.
    - **Hemoglobin:** Low levels (anemia) often accompany CKD.
    - **Blood Glucose:** Diabetes is a leading CKD cause; monitor for hyperglycemia.

    **Model Interpretation:**
    This Random Forest model aggregates feature importance to predict CKD (1: Yes, 0: No). Confidence scores reflect prediction reliability.
    """)

    # Example DataFrame for interpretive table
    interpretive_data = {
        "Biomarker": ["Hemoglobin", "Serum Creatinine", "Blood Urea", "Blood Glucose"],
        "Normal Range": ["12-16 g/dL", "0.6-1.2 mg/dL", "7-20 mg/dL", "<140 mg/dL"],
        "Risk if Abnormal": ["Low (Anemia)", "High (Poor Filtration)", "High (Kidney Stress)", "High (Diabetes Link)"],
        "Weight in Model": ["Medium", "High", "High", "Medium"]
    }
    df = pd.DataFrame(interpretive_data)
    st.table(df)

    st.markdown("""
    **Disclaimer:** This tool is for informational purposes. Always seek professional medical advice.
    """)

with tab3:
    st.markdown('<h1 class="main-header">Batch Benchmark</h1>', unsafe_allow_html=True)

    st.markdown("""
    Run a batch of predictions to evaluate model performance under load. Uses the current input data as sample.
    """)

    n_runs = st.slider("Number of Runs", min_value=10, max_value=500, value=100, step=10)

    if st.button("🚀 Run Benchmark", type="primary"):
        input_data = st.session_state.get('input_data', default_input_data)
        with st.spinner(f"Running {n_runs} predictions..."):
            samples = [input_data] * 10  # Simulate multiple patients
            stats = batch_benchmark(samples, st.session_state.model, n_runs=n_runs)
            if stats:
                snap = system_snapshot()

                # Benchmark Metrics
                st.subheader("📊 Benchmark Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="metric-card">**⏱️ Latency Metrics (ms)**</div>', unsafe_allow_html=True)
                    st.metric("P50 (Median)", stats["p50_ms"])
                    st.metric("P95", stats["p95_ms"])
                    st.metric("Mean", stats["mean_ms"])

                with col2:
                    st.markdown('<div class="metric-card">**⚡ Throughput**</div>', unsafe_allow_html=True)
                    if stats["throughput_rps"] == float('inf'):
                        st.metric("Requests/Second", ">10000")  # Indicate very high
                    else:
                        st.metric("Requests/Second", stats["throughput_rps"])

                # System Snapshot
                st.subheader("💻 System Snapshot During Benchmark")
                col_sys1, col_sys2 = st.columns(2)
                with col_sys1:
                    st.metric("CPU Usage", f"{snap['cpu_percent']:.1f}%")
                    st.metric("RAM Used", f"{snap['ram_used_mb']} MB")
                with col_sys2:
                    st.metric("Total RAM", f"{snap['ram_total_mb']} MB")
                    with st.expander("Details"):
                        st.write(f"**OS:** {snap['os']}")
                        st.write(f"**Processor:** {snap['processor']}")
            else:
                st.error("Benchmark failed. Ensure model is loaded.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #808080;'>© 2025 NefroAi | Powered by Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)
