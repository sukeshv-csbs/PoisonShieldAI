import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

from dataset import load_uploaded_mnist, create_poisoned_dataset
from detector import calculate_risk_scores
from model import train_model, evaluate_model, train_cleaned_model


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PoisonShield AI",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* GLOBAL SKY BLUE TEXT */
:root {
    --sky-blue: #7dd3fc;
    --sky-blue-bright: #38bdf8;
}

/* MAIN BACKGROUND */

.stApp {
    background: #0b1120;
    color: #7dd3fc !important;
}

/* GLOBAL TEXT */

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp small,
.stApp strong,
.stApp b,
.stApp div,
.stApp summary {
    color: #7dd3fc !important;
}

/* MARKDOWN TEXT */

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
.stMarkdown strong {
    color: #7dd3fc !important;
    line-height: 1.65;
}

.stMarkdown p {
    background: transparent;
    border: none;
    padding: 0;
    margin-bottom: 10px;
}

.stMarkdown ul,
.stMarkdown ol {
    background: transparent;
    border: none;
    padding-left: 25px;
}

/* HEADINGS */

h1, h2, h3, h4, h5, h6,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #7dd3fc !important;
    font-weight: 700 !important;
}

/* MAIN CONTAINER */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* CAPTIONS */

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #7dd3fc !important;
    font-size: 0.85rem;
}

/* HEADER */

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #7dd3fc !important;
    letter-spacing: -1px;
}

.main-subtitle {
    color: #7dd3fc !important;
    font-size: 1rem;
    margin-top: 8px;
    margin-bottom: 1.5rem;
}

/* ==========================================
   CUSTOM MESSAGE CARDS
   ========================================== */

[data-testid="stAlert"] {
    background: linear-gradient(
        135deg,
        #142b42,
        #172c43
    ) !important;
    border: 1px solid #2584a8 !important;
    border-left: 4px solid #22d3ee !important;
    border-radius: 12px !important;
    padding: 13px 17px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #7dd3fc !important;
    background: transparent !important;
    border: none !important;
    font-weight: 500;
}

[data-testid="stAlert"] svg {
    color: #38bdf8 !important;
    fill: none;
}

/* ==========================================
   METRIC CARDS
   ========================================== */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172337, #111c2e);
    border: 1px solid #29415e;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #7dd3fc !important;
    font-weight: 700;
}

/* ==========================================
   BUTTONS
   ========================================== */

.stButton > button {
    background: linear-gradient(135deg, #0891b2, #2563eb);
    color: #7dd3fc !important;
    border: 1px solid #38bdf8;
    border-radius: 10px;
    min-height: 44px;
    padding: 0.6rem 1.3rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    box-shadow: 0 4px 12px rgba(8, 145, 178, 0.18);
    transition: all 0.2s ease;
}

.stButton > button *,
.stButton > button p,
.stButton > button span {
    color: #7dd3fc !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    border-color: #7dd3fc;
    box-shadow: 0 5px 18px rgba(56, 189, 248, 0.3);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: scale(0.98);
}

/* ==========================================
   FILE UPLOADER
   ========================================== */

[data-testid="stFileUploader"] {
    background: #142238;
    border: 1.5px dashed #38bdf8;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
    border: none;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #0891b2, #2563eb) !important;
    color: #7dd3fc !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button span {
    color: #7dd3fc !important;
}

[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
}

/* FILE UPLOADER TEXT */

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #7dd3fc !important;
}

/* ==========================================
   DOWNLOAD BUTTON
   ========================================== */

.stDownloadButton > button {
    background: linear-gradient(135deg, #0891b2, #2563eb);
    color: #7dd3fc !important;
    border: 1px solid #38bdf8;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stDownloadButton > button *,
.stDownloadButton > button span {
    color: #7dd3fc !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    border-color: #7dd3fc;
    box-shadow: 0 5px 18px rgba(56, 189, 248, 0.3);
    transform: translateY(-1px);
}

/* ==========================================
   TABS
   ========================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #263b55;
}

.stTabs [data-baseweb="tab"] {
    background: #111c2e;
    border-radius: 8px 8px 0 0;
    color: #7dd3fc !important;
    padding: 10px 15px;
}

.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] p {
    color: #7dd3fc !important;
}

.stTabs [aria-selected="true"] {
    background: #172c43 !important;
    color: #7dd3fc !important;
}

/* ==========================================
   SIDEBAR
   ========================================== */

section[data-testid="stSidebar"] {
    background: #101b2d;
    border-right: 1px solid #263b55;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {
    color: #7dd3fc !important;
}

/* ==========================================
   INPUTS
   ========================================== */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #172337 !important;
    color: #7dd3fc !important;
    border: 1px solid #29415e;
    border-radius: 8px;
    caret-color: #7dd3fc;
}

.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #7dd3fc !important;
    opacity: 0.7;
}

[data-baseweb="select"] > div {
    background: #172337;
    border-color: #29415e;
    color: #7dd3fc !important;
}

[data-baseweb="select"] span {
    color: #7dd3fc !important;
}

/* INPUT LABELS */

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stTextArea label {
    color: #7dd3fc !important;
}

/* ==========================================
   WORKFLOW CARDS
   ========================================== */

.workflow-card {
    background: linear-gradient(145deg, #172c43, #111c2e);
    border: 1px solid #2584a8;
    border-radius: 14px;
    padding: 20px;
    min-height: 155px;
    margin-bottom: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

.workflow-number {
    color: #7dd3fc !important;
    font-size: 1.4rem;
    font-weight: 800;
}

.workflow-card h4 {
    color: #7dd3fc !important;
    margin-top: 8px;
}

.workflow-text {
    color: #7dd3fc !important;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ==========================================
   DATAFRAMES
   ========================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #29415e;
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    color: #7dd3fc !important;
}

/* ==========================================
   SLIDER
   ========================================== */

[data-testid="stSlider"] * {
    color: #7dd3fc !important;
}

/* ==========================================
   PROGRESS BAR
   ========================================== */

[data-testid="stProgress"] > div {
    background: #172337;
}

[data-testid="stProgress"] [data-testid="stProgressBar"] {
    background: #38bdf8;
}

/* ==========================================
   DIVIDERS
   ========================================== */

hr {
    border-color: #263b55 !important;
}

/* ==========================================
   FOOTER
   ========================================== */

.custom-footer {
    color: #7dd3fc !important;
    text-align: center;
    padding: 18px;
    font-size: 0.85rem;
    background: linear-gradient(
        135deg,
        #142b42,
        #172c43
    );
    border: 1px solid #29415e;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# CUSTOM MESSAGE FUNCTIONS
# ==========================================

def info_card(message):
    st.info(message)


def success_card(message):
    st.success(message)


def warning_card(message):
    st.warning(message)


def error_card(message):
    st.error(message)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="main-title">🛡️ PoisonShield AI</div>
<div class="main-subtitle">
Explainable AI-based training dataset security scanner
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "analysis": None,
    "original_model": None,
    "cleaned_model": None,
    "cleaned_results": None,
    "experiment_results": None,
    "reviewed_indices": [],
    "uploaded_data": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_sample_losses(model, images, labels):

    probabilities = model.predict(
        images,
        batch_size=256,
        verbose=0
    )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0
    )

    return -np.log(
        probabilities[np.arange(len(labels)), labels]
    )


# ==========================================
# DATASET SCANNING
# ==========================================

def run_dataset_scan(
    x_train,
    y_train,
    max_samples=6000,
    threshold=95,
    random_state=2026
):

    total = len(x_train)

    rng = np.random.default_rng(random_state)

    if total > max_samples:

        selected_indices = rng.choice(
            total,
            size=max_samples,
            replace=False
        )

        selected_indices.sort()

    else:

        selected_indices = np.arange(total)

    x_scan = x_train[selected_indices]
    y_scan = y_train[selected_indices]

    if len(x_scan) < 6:
        raise ValueError(
            "At least six samples are required for scanning."
        )

    indices = rng.permutation(len(x_scan))

    folds = np.array_split(indices, 3)

    losses = np.zeros(
        len(x_scan),
        dtype=np.float32
    )

    progress = st.progress(0)
    status = st.empty()

    for fold_number in range(3):

        validation_indices = folds[fold_number]

        training_indices = np.concatenate([
            folds[i]
            for i in range(3)
            if i != fold_number
        ])

        status.write(
            f"Scanning fold {fold_number + 1} of 3..."
        )

        model, _ = train_model(
            x_scan[training_indices],
            y_scan[training_indices],
            epochs=2
        )

        fold_losses = calculate_sample_losses(
            model,
            x_scan[validation_indices],
            y_scan[validation_indices]
        )

        losses[validation_indices] = fold_losses

        progress.progress(
            (fold_number + 1) / 3
        )

    progress.empty()
    status.empty()

    risk_scores = calculate_risk_scores(losses)

    suspicious_local = np.where(
        risk_scores >= threshold
    )[0]

    suspicious_global = selected_indices[
        suspicious_local
    ]

    status.write(
        "Training reference model on the uploaded dataset..."
    )

    original_model, history = train_model(
        x_train,
        y_train,
        epochs=3
    )

    status.empty()

    return {
        "losses": losses,
        "risk_scores": risk_scores,
        "selected_indices": selected_indices,
        "suspicious_indices": suspicious_global,
        "original_model": original_model,
        "history": history
    }


# ==========================================
# CONTROLLED POISONING EXPERIMENT
# ==========================================

def run_poisoning_experiment(
    x_train,
    y_train,
    threshold=95,
    flip_rate=0.10
):

    poisoned = create_poisoned_dataset(
        x_train,
        y_train,
        flip_rate=flip_rate,
        random_state=42
    )

    poisoned_labels = poisoned["y_train_poisoned"]
    actual_poisoned = poisoned["poisoned_indices"]

    scan = run_dataset_scan(
        x_train,
        poisoned_labels,
        max_samples=6000,
        threshold=threshold,
        random_state=2026
    )

    scanned_indices = scan["selected_indices"]
    suspicious_indices = scan["suspicious_indices"]

    actual_set = set(actual_poisoned.tolist())
    detected_set = set(suspicious_indices.tolist())
    scanned_set = set(scanned_indices.tolist())

    actual_scanned = actual_set.intersection(scanned_set)
    detected_scanned = detected_set.intersection(scanned_set)

    true_positive = len(
        actual_scanned.intersection(detected_scanned)
    )

    false_positive = len(
        detected_scanned - actual_scanned
    )

    false_negative = len(
        actual_scanned - detected_scanned
    )

    true_negative = (
        len(scanned_set)
        - true_positive
        - false_positive
        - false_negative
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "poisoned": poisoned,
        "suspicious_indices": suspicious_indices,
        "actual_poisoned": actual_poisoned,
        "scanned_indices": scanned_indices,
        "actual_scanned_count": len(actual_scanned),
        "detected_poisoned_count": true_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "scan": scan
    }


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("## 🛡️ PoisonShield AI")
st.sidebar.caption("DATASET SECURITY SCANNER")
st.sidebar.markdown("---")

st.sidebar.header("📂 Dataset Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload your MNIST dataset files",
    type=["npz", "csv", "idx", "ubyte", "gz"],
    accept_multiple_files=True
)

st.sidebar.caption(
    "Upload one NPZ file, one CSV file, "
    "or matching IDX/UBYTE image and label files."
)

if st.sidebar.button(
    "Load Dataset",
    type="primary",
    use_container_width=True
):

    if not uploaded_files:

        st.sidebar.error(
            "Please upload dataset files first."
        )

    else:

        try:

            with st.spinner(
                "Loading and validating dataset..."
            ):

                data = load_uploaded_mnist(uploaded_files)

                x_train, y_train, x_test, y_test = data

                st.session_state.uploaded_data = {
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_test": x_test,
                    "y_test": y_test
                }

                st.session_state.analysis = None
                st.session_state.original_model = None
                st.session_state.cleaned_model = None
                st.session_state.cleaned_results = None
                st.session_state.experiment_results = None
                st.session_state.reviewed_indices = []

            st.sidebar.success(
                "Dataset loaded successfully!"
            )

        except Exception as e:

            st.sidebar.error(str(e))


# ==========================================
# WELCOME PAGE
# ==========================================

data = st.session_state.uploaded_data

if data is None:

    st.markdown("### Welcome to PoisonShield AI")

    st.markdown("""
    PoisonShield AI helps identify potentially suspicious samples
    in machine-learning training datasets.

    Upload your dataset to start the security analysis.
    """)

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-number">01</div>
            <h4>Upload Dataset</h4>
            <div class="workflow-text">
                Upload and validate your MNIST-compatible training data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-number">02</div>
            <h4>Detect Suspicious Data</h4>
            <div class="workflow-text">
                Analyze samples using model prediction loss and risk scores.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="workflow-card">
            <div class="workflow-number">03</div>
            <h4>Review and Clean</h4>
            <div class="workflow-text">
                Review suspicious samples and retrain using selected data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    info_card(
        "Upload your dataset using the sidebar and click Load Dataset."
    )

    st.stop()


# ==========================================
# DATASET INFORMATION
# ==========================================

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

st.markdown("### Dataset Overview")

success_card(
    "Dataset successfully loaded and validated."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Training Samples", f"{len(x_train):,}")
col2.metric("Testing Samples", f"{len(x_test):,}")
col3.metric("Image Size", "28 × 28")
col4.metric("Classes", "10")


# ==========================================
# TABS
# ==========================================

tabs = st.tabs([
    "📊 Dashboard",
    "🚨 Suspicious Samples",
    "🧹 User Review",
    "⚖️ Model Comparison",
    "🔍 Sample Inspector",
    "🧪 Poisoning Experiment"
])


# ==========================================
# TAB 1: DASHBOARD
# ==========================================

with tabs[0]:

    st.subheader("Dataset Security Dashboard")

    st.write(
        "Run the scanner to identify samples with unusually high "
        "prediction loss."
    )

    info_card(
        "Automatic detection uses the 95th-percentile risk threshold."
    )

    if st.button("🚀 Run Dataset Scan", type="primary"):

        try:

            with st.spinner(
                "Scanning dataset and training reference model..."
            ):

                result = run_dataset_scan(
                    x_train,
                    y_train
                )

                st.session_state.analysis = result
                st.session_state.original_model = result["original_model"]
                st.session_state.cleaned_model = None
                st.session_state.cleaned_results = None
                st.session_state.reviewed_indices = []

            success_card("Dataset scan completed!")

        except Exception as e:

            error_card(f"Scan failed: {e}")

    analysis = st.session_state.analysis

    if analysis is not None:

        suspicious = analysis["suspicious_indices"]
        scores = analysis["risk_scores"]

        a, b, c = st.columns(3)

        a.metric(
            "Samples Scanned",
            f"{len(analysis['selected_indices']):,}"
        )

        b.metric(
            "Suspicious Samples",
            f"{len(suspicious):,}"
        )

        c.metric(
            "Suspicious Rate",
            f"{len(suspicious) / len(scores) * 100:.2f}%"
        )

        st.markdown("#### Risk Score Distribution")

        fig, ax = plt.subplots()

        fig.patch.set_facecolor("#111c2e")
        ax.set_facecolor("#172337")

        ax.hist(scores, bins=30, edgecolor="#0b1120")

        ax.axvline(
            95,
            linestyle="--",
            label="95th percentile threshold"
        )

        ax.set_xlabel("Relative Risk Score", color="#7dd3fc")
        ax.set_ylabel("Number of Samples", color="#7dd3fc")
        ax.set_title("Sample Risk Distribution", color="#7dd3fc")

        ax.tick_params(colors="#7dd3fc")
        ax.legend()

        for spine in ax.spines.values():
            spine.set_color("#29415e")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Risk scores are relative percentile ranks, not probabilities "
            "that a sample is poisoned. The scanner flags samples at or "
            "above the 95th-percentile threshold."
        )

        st.markdown("#### Dataset Class Distribution")

        class_counts = (
            pd.Series(y_train)
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots()

        fig.patch.set_facecolor("#111c2e")
        ax.set_facecolor("#172337")

        ax.bar(
            class_counts.index.astype(str),
            class_counts.values
        )

        ax.set_xlabel("Digit Class", color="#7dd3fc")
        ax.set_ylabel("Number of Samples", color="#7dd3fc")

        ax.set_title(
            "Training Dataset Class Distribution",
            color="#7dd3fc"
        )

        ax.tick_params(colors="#7dd3fc")

        for spine in ax.spines.values():
            spine.set_color("#29415e")

        st.pyplot(fig)
        plt.close(fig)


# ==========================================
# TAB 2: SUSPICIOUS SAMPLES
# ==========================================

with tabs[1]:

    st.subheader("Suspicious Sample Detection")

    analysis = st.session_state.analysis

    if analysis is None:

        warning_card("Run the dataset scan first.")

    else:

        suspicious = analysis["suspicious_indices"]
        selected_indices = analysis["selected_indices"]
        scores = analysis["risk_scores"]

        score_lookup = {
            int(index): float(score)
            for index, score in zip(selected_indices, scores)
        }

        if len(suspicious) == 0:

            success_card("No samples were selected.")

        else:

            st.write(
                f"Found **{len(suspicious)}** samples for review."
            )

            rows = []

            for index in suspicious:

                rows.append({
                    "Sample Index": int(index),
                    "Actual Label": int(y_train[index]),
                    "Risk Score": round(
                        score_lookup[int(index)], 2
                    )
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True
            )

            info_card(
                "High risk indicates unusual model loss. "
                "It does not prove that a sample is poisoned."
            )


# ==========================================
# TAB 3: USER REVIEW
# ==========================================

with tabs[2]:

    st.subheader("Human Review and Dataset Cleaning")

    analysis = st.session_state.analysis

    if analysis is None:

        warning_card("Run the dataset scan first.")

    else:

        suspicious = analysis["suspicious_indices"]

        if len(suspicious) == 0:

            info_card("There are no samples to review.")

        else:

            st.write(
                "Choose the samples you want to remove. "
                "Nothing is removed automatically."
            )

            options = [int(i) for i in suspicious]

            selected_for_removal = st.multiselect(
                "Select samples to remove",
                options=options,
                default=st.session_state.reviewed_indices,
                format_func=lambda i: (
                    f"Sample {i} — Label {int(y_train[i])}"
                )
            )

            st.session_state.reviewed_indices = selected_for_removal

            if selected_for_removal:

                st.markdown("#### Selected Sample Preview")

                preview_index = selected_for_removal[0]

                fig, ax = plt.subplots()

                fig.patch.set_facecolor("#111c2e")
                ax.set_facecolor("#172337")

                ax.imshow(
                    x_train[preview_index],
                    cmap="gray"
                )

                ax.set_title(
                    f"Sample {preview_index} | "
                    f"Label: {y_train[preview_index]}",
                    color="#7dd3fc"
                )

                ax.axis("off")

                st.pyplot(fig)
                plt.close(fig)

            st.write(
                f"Selected for removal: **{len(selected_for_removal)}**"
            )

            if st.button(
                "🧹 Clean Dataset and Retrain Model",
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Removing selected samples and retraining..."
                    ):

                        (
                            cleaned_model,
                            history,
                            x_clean,
                            y_clean
                        ) = train_cleaned_model(
                            x_train,
                            y_train,
                            selected_for_removal,
                            epochs=3
                        )

                        test_loss, test_accuracy = evaluate_model(
                            cleaned_model,
                            x_test,
                            y_test
                        )

                        st.session_state.cleaned_model = cleaned_model

                        st.session_state.cleaned_results = {
                            "history": history,
                            "x_clean": x_clean,
                            "y_clean": y_clean,
                            "test_loss": test_loss,
                            "test_accuracy": test_accuracy,
                            "removed_count": len(selected_for_removal)
                        }

                    success_card(
                        "Cleaning and retraining completed!"
                    )

                except Exception as e:

                    error_card(f"Cleaning failed: {e}")

            if st.session_state.cleaned_results is not None:

                results = st.session_state.cleaned_results

                st.markdown("#### Cleaning Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Remaining Samples",
                    f"{len(results['x_clean']):,}"
                )

                col2.metric(
                    "Removed Samples",
                    results["removed_count"]
                )

                col3.metric(
                    "Cleaned Model Accuracy",
                    f"{results['test_accuracy'] * 100:.2f}%"
                )

                buffer = io.BytesIO()

                np.savez_compressed(
                    buffer,
                    x_train=results["x_clean"],
                    y_train=results["y_clean"],
                    x_test=x_test,
                    y_test=y_test
                )

                st.download_button(
                    "⬇️ Download Cleaned Dataset",
                    data=buffer.getvalue(),
                    file_name="cleaned_dataset.npz",
                    mime="application/octet-stream"
                )


# ==========================================
# TAB 4: MODEL COMPARISON
# ==========================================

with tabs[3]:

    st.subheader("Original vs Cleaned Model")

    original_model = st.session_state.original_model
    cleaned_model = st.session_state.cleaned_model

    if original_model is None:

        warning_card("Run the dataset scan first.")

    elif cleaned_model is None:

        info_card(
            "Review suspicious samples and retrain the cleaned model "
            "to compare both models."
        )

    else:

        try:

            original_loss, original_accuracy = evaluate_model(
                original_model,
                x_test,
                y_test
            )

            cleaned_loss, cleaned_accuracy = evaluate_model(
                cleaned_model,
                x_test,
                y_test
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("#### Original Model")

                st.metric(
                    "Test Accuracy",
                    f"{original_accuracy * 100:.2f}%"
                )

                st.metric(
                    "Test Loss",
                    f"{original_loss:.4f}"
                )

            with col2:

                st.markdown("#### Cleaned Model")

                st.metric(
                    "Test Accuracy",
                    f"{cleaned_accuracy * 100:.2f}%",
                    delta=(
                        f"{(cleaned_accuracy - original_accuracy) * 100:.2f}%"
                    )
                )

                st.metric(
                    "Test Loss",
                    f"{cleaned_loss:.4f}"
                )

            comparison = pd.DataFrame({
                "Model": ["Original", "Cleaned"],
                "Accuracy": [
                    original_accuracy * 100,
                    cleaned_accuracy * 100
                ]
            })

            fig, ax = plt.subplots()

            fig.patch.set_facecolor("#111c2e")
            ax.set_facecolor("#172337")

            ax.bar(
                comparison["Model"],
                comparison["Accuracy"]
            )

            ax.set_ylabel(
                "Test Accuracy (%)",
                color="#7dd3fc"
            )

            ax.set_ylim(0, 100)

            ax.set_title(
                "Model Accuracy Comparison",
                color="#7dd3fc"
            )

            ax.tick_params(colors="#7dd3fc")

            for spine in ax.spines.values():
                spine.set_color("#29415e")

            st.pyplot(fig)
            plt.close(fig)

            info_card(
                "Both models are evaluated on the same held-out test set. "
                "Cleaning does not guarantee an improvement in accuracy."
            )

        except Exception as e:

            error_card(f"Comparison failed: {e}")


# ==========================================
# TAB 5: SAMPLE INSPECTOR
# ==========================================

with tabs[4]:

    st.subheader("Individual Sample Inspector")

    sample_index = st.number_input(
        "Enter sample index",
        min_value=0,
        max_value=len(x_train) - 1,
        value=0,
        step=1
    )

    sample_index = int(sample_index)

    fig, ax = plt.subplots()

    fig.patch.set_facecolor("#111c2e")
    ax.set_facecolor("#172337")

    ax.imshow(
        x_train[sample_index],
        cmap="gray"
    )

    ax.set_title(
        f"Training Sample {sample_index}",
        color="#7dd3fc"
    )

    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    col1.metric(
        "Actual Label",
        int(y_train[sample_index])
    )

    if st.session_state.original_model is not None:

        prediction = st.session_state.original_model.predict(
            x_train[sample_index:sample_index + 1],
            verbose=0
        )[0]

        predicted_label = int(np.argmax(prediction))

        col2.metric(
            "Model Prediction",
            predicted_label
        )

        st.write(
            f"Prediction confidence: "
            f"**{np.max(prediction) * 100:.2f}%**"
        )

        if predicted_label != int(y_train[sample_index]):

            warning_card(
                "The model prediction differs from the supplied label. "
                "This alone does not prove poisoning."
            )

        else:

            success_card(
                "The model prediction matches the supplied label."
            )

    else:

        col2.metric(
            "Model Prediction",
            "Not scanned yet"
        )


# ==========================================
# TAB 6: POISONING EXPERIMENT
# ==========================================

with tabs[5]:

    st.subheader("Controlled Label-Flipping Experiment")

    st.write(
        "This experiment deliberately changes a portion of training "
        "labels to test the scanner. It does not modify your uploaded "
        "dataset."
    )

    flip_rate = st.slider(
        "Label-flipping rate (%)",
        min_value=1,
        max_value=30,
        value=10,
        key="flip_rate_slider"
    )

    info_card(
        "The scanner automatically uses the 95th-percentile "
        "risk threshold to flag suspicious samples."
    )

    if st.button(
        "🧪 Run Poisoning Experiment",
        type="primary"
    ):

        try:

            with st.spinner(
                "Creating controlled poisoning experiment and scanning..."
            ):

                results = run_poisoning_experiment(
                    x_train,
                    y_train,
                    flip_rate=flip_rate / 100
                )

                st.session_state.experiment_results = results

            success_card("Experiment completed!")

        except Exception as e:

            error_card(f"Experiment failed: {e}")

    results = st.session_state.experiment_results

    if results is not None:

        st.markdown("#### Detection Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{results['precision'] * 100:.2f}%"
        )

        col2.metric(
            "Recall",
            f"{results['recall'] * 100:.2f}%"
        )

        col3.metric(
            "F1 Score",
            f"{results['f1'] * 100:.2f}%"
        )

        st.markdown("#### Confusion Matrix")

        matrix = pd.DataFrame(
            [
                [
                    results["true_negative"],
                    results["false_positive"]
                ],
                [
                    results["false_negative"],
                    results["true_positive"]
                ]
            ],
            index=[
                "Actually Clean",
                "Actually Poisoned"
            ],
            columns=[
                "Detected Clean",
                "Detected Suspicious"
            ]
        )

        st.dataframe(
            matrix,
            use_container_width=True
        )

        st.markdown("#### Poisoning Summary")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Training Samples",
                f"{len(x_train):,}"
            )

            st.metric(
                "Injected Poisoned Samples",
                f"{len(results['actual_poisoned']):,}"
            )

            st.metric(
                "Samples Evaluated",
                f"{len(results['scanned_indices']):,}"
            )

        with col2:

            st.metric(
                "Poisoned Samples in Scan",
                f"{results['actual_scanned_count']:,}"
            )

            st.metric(
                "Suspicious Samples Detected",
                f"{len(results['suspicious_indices']):,}"
            )

            st.metric(
                "Correctly Detected",
                f"{results['detected_poisoned_count']:,}"
            )

        info_card(
            "Metrics are calculated only for the samples actually scanned. "
            "Precision, recall and F1 are valid for this controlled "
            "label-flipping experiment, where injected indices are known. "
            "Real uploaded datasets do not provide poisoning ground truth "
            "automatically."
        )

        st.markdown("#### Class Distribution After Label Flipping")

        report = results["poisoned"]["poisoning_report"]

        st.dataframe(
            report,
            use_container_width=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="custom-footer">
🛡️ <b>PoisonShield AI</b><br>
Explainable Dataset Security Research Prototype<br>
Human-Reviewed Dataset Cleaning | MNIST-Compatible Data
</div>
""", unsafe_allow_html=True)