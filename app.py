import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG + STYLE
# =========================
st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
/* Biar padding agak lega */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Card look */
.card {
    padding: 1.25rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.small-note { font-size: 0.9rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_bundle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

bundle = load_bundle("final_logreg_threshold.pkl")

# validasi tipe
if isinstance(bundle, dict) and "model" in bundle:
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))  # fallback aman
else:
    # fallback kalau ternyata yg keload masih model/pipeline lama
    model = bundle
    threshold = 0.5
    st.warning("Threshold tidak ditemukan di file model. Menggunakan default threshold=0.5.")


# =========================
# HEADER
# =========================
st.title("🏦 Bank Term Deposit Prediction")
st.write("Aplikasi ini memprediksi apakah nasabah berpotensi berlangganan deposito berjangka (**Yes/No**).")

# =========================
# LAYOUT
# =========================
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Input Data Nasabah")

    tab1, tab2, tab3 = st.tabs(["Profil", "Kontak & Campaign", "Indikator Makro"])

    # ---- TAB 1: Profil
    with tab1:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            age = st.number_input("Age", min_value=17, max_value=100, value=35, step=1)

            job = st.selectbox("Job", [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown"
            ])

            marital = st.selectbox("Marital", ["married", "single", "divorced", "unknown"])

        with c2:
            education = st.selectbox("Education", [
                "basic", "high.school", "professional.course", "university.degree",
                "illiterate", "unknown"
            ])

            default = st.selectbox("Default", ["no", "yes", "unknown"])
            housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
            loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])

    # ---- TAB 2: Kontak & Campaign
    with tab2:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            contact = st.selectbox("Contact", ["cellular", "telephone"])
            month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
            day_of_week = st.selectbox("Day of Week", ["mon","tue","wed","thu","fri"])

        with c2:
            campaign = st.number_input("Campaign (No. of contacts)", min_value=1, value=1, step=1)
            pdays = st.number_input("Pdays (days since last contact)", min_value=0, value=0, step=1)
            previous = st.number_input("Previous (contacts before campaign)", min_value=0, value=0, step=1)

        poutcome = st.selectbox("Poutcome (previous outcome)", ["nonexistent", "failure", "success"])

    # ---- TAB 3: Makro
    with tab3:
        st.caption("Indikator ekonomi yang mempengaruhi keputusan nasabah.")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            cons_price_idx = st.number_input("cons.price.idx", value=93.994)
        with c2:
            cons_conf_idx = st.number_input("cons.conf.idx", value=-36.4)
        with c3:
            euribor3m = st.number_input("euribor3m", value=4.857)

    st.markdown("---")

    # threshold (opsional) untuk adjust recall/precision
#    st.subheader("⚙️ Pengaturan (Opsional)")
#    threshold = st.slider(
#        "Decision Threshold untuk YES",
#        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
#        help="Turunkan threshold untuk menaikkan recall (lebih banyak YES terdeteksi), tapi biasanya precision turun."
#    )

    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PREPARE INPUT DATA
# =========================
input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "month": [month],
    "day_of_week": [day_of_week],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome],
    "cons.price.idx": [cons_price_idx],
    "cons.conf.idx": [cons_conf_idx],
    "euribor3m": [euribor3m],
})

# =========================
# OUTPUT PANEL
# =========================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Hasil Prediksi")

    st.caption("Klik **Prediksi Sekarang** untuk melihat hasil dan probabilitas.")

    # preview input ringkas
    with st.expander("Lihat ringkasan input", expanded=False):
        st.dataframe(input_data, use_container_width=True)

    if predict_btn:
        # predict proba
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_data)[0, 1])
            pred = int(proba >= threshold)
        else:
            pred = int(model.predict(input_data)[0])

        # UI output
        if pred == 1:
            st.success("Prediksi: **YES — Subscribe Deposit** ✅")
        else:
            st.info("Prediksi: **NO — Not Subscribe**")

        # metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Decision", "YES" if pred == 1 else "NO")
        with m2:
            if proba is not None:
                st.metric("Probabilitas YES", f"{proba:.3f}")

        if proba is not None:
            st.write("Confidence (YES)")
            st.progress(min(max(proba, 0.0), 1.0))

        st.markdown("---")
        st.caption("Model: Logistic Regression (Tuned Pipeline)")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")