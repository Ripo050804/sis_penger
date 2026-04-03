"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Fitur: Kamera, Upload, Info Kelas, Download PDF, Mobile-friendly
Dengan auto-download model dari Google Drive
Versi: Stabil untuk Streamlit Cloud (Python 3.14 compatible)
Kelas: Arca, dolmen, menhir, dakon, batu_datar, Kubur_batu, Lesung_batu
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import os
from fpdf import FPDF
import gdown
import requests

# ==============================================
# KONFIGURASI HALAMAN
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================
# DESKRIPSI KELAS
# ==============================================
DESKRIPSI_KELAS = {
    "Arca": "Arca adalah patung batu yang melambangkan nenek moyang atau dewa. Biasanya berbentuk manusia atau hewan dengan pahatan detail pada permukaan batu, dan ditemukan di situs megalitik sebagai objek pemujaan.",
    "dolmen": "Dolmen adalah meja batu yang terdiri dari beberapa batu tegak yang menopang batu datar di atasnya. Digunakan sebagai tempat meletakkan sesaji atau untuk upacara ritual pada masa prasejarah.",
    "menhir": "Menhir adalah tugu batu tegak yang didirikan sebagai tanda peringatan atau simbol kekuatan. Biasanya berbentuk memanjang vertikal dan ditemukan berdiri sendiri atau berkelompok di situs megalitik.",
    "dakon": "Dakon adalah batu berlubang-lubang kecil yang tersusun pada permukaannya menyerupai papan permainan congkak. Diduga digunakan untuk ritual keagamaan atau permainan tradisional pada masa megalitikum.",
    "batu_datar": "Batu datar adalah batu besar berbentuk lempengan dengan permukaan yang relatif rata. Mungkin digunakan sebagai alas, tempat duduk, atau altar dalam upacara adat pada masa prasejarah.",
    "Kubur_batu": "Kubur batu adalah peti mati yang terbuat dari batu, digunakan untuk mengubur jenazah pada masa megalitik. Biasanya terdiri dari susunan batu besar yang membentuk ruang atau wadah.",
    "Lesung_batu": "Lesung batu adalah artefak batu yang memiliki cekungan besar pada bagian permukaannya. Digunakan sebagai alat tradisional untuk menumbuk atau menghaluskan bahan makanan pada masa prasejarah."
}

# ==============================================
# FILE_ID MODEL TFLITE DI GOOGLE DRIVE
# ==============================================
FILE_ID = "1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx"

# ==============================================
# FUNGSI LOAD TFLITE INTERPRETER
# (kompatibel tflite-runtime & tensorflow)
# ==============================================
def load_tflite_interpreter(model_path):
    # Coba tflite-runtime dulu (lebih ringan)
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except Exception:
        pass
    # Fallback ke tensorflow
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except Exception:
        pass
    return None


# ==============================================
# FUNGSI DOWNLOAD & LOAD MODEL
# ==============================================
@st.cache_resource
def download_and_load_model():
    model_path = "megalitikum_model.tflite"

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
        with st.spinner("🔄 Mendownload model (±96 MB)... Harap tunggu."):
            # Metode 1: gdown
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={FILE_ID}",
                    model_path, quiet=False, fuzzy=True
                )
                if os.path.exists(model_path) and os.path.getsize(model_path) > 10000:
                    st.success("✅ Model berhasil didownload!")
                else:
                    raise Exception("File terlalu kecil")
            except Exception:
                # Metode 2: requests
                try:
                    st.warning("⏳ Mencoba metode alternatif...")
                    session = requests.Session()
                    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
                    resp = session.get(url, stream=True, timeout=120)
                    for k, v in resp.cookies.items():
                        if k.startswith('download_warning'):
                            url = f"https://drive.google.com/uc?export=download&confirm={v}&id={FILE_ID}"
                            resp = session.get(url, stream=True, timeout=120)
                            break
                    with open(model_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)
                    if not (os.path.exists(model_path) and os.path.getsize(model_path) > 10000):
                        raise Exception("Download gagal")
                    st.success("✅ Model berhasil didownload!")
                except Exception as e2:
                    st.error(f"❌ Gagal download: {str(e2)}")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    return None, None, None

    interp = load_tflite_interpreter(model_path)
    if interp is None:
        st.error("❌ Tidak dapat memuat model TFLite. Periksa library di requirements.txt.")
        return None, None, None

    return interp, interp.get_input_details(), interp.get_output_details()


# ==============================================
# FUNGSI ENHANCEMENT GAMBAR
# ==============================================
def adaptive_enhancement(image, brightness, contrast):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.filter(ImageFilter.SHARPEN)
    if contrast < 40:
        image = ImageEnhance.Contrast(image).enhance(1.5)
    if brightness < 80:
        image = ImageEnhance.Brightness(image).enhance(1.3)
    elif brightness > 180:
        image = ImageEnhance.Brightness(image).enhance(0.8)
    return image


# ==============================================
# FUNGSI DETEKSI NON-BATU
# ==============================================
def detect_non_megalith(image):
    try:
        img = image.convert('RGB')
        r, g, b = img.split()
        rm = np.mean(np.array(r))
        gm = np.mean(np.array(g))
        bm = np.mean(np.array(b))
        if gm > rm * 1.2 and gm > bm * 1.2:
            return True, "Gambar didominasi warna hijau (mungkin tumbuhan)"
        if bm > rm * 1.3 and bm > gm * 1.3:
            return True, "Gambar didominasi warna biru (mungkin langit/air)"
        if np.var([rm, gm, bm]) > 500:
            return True, "Variasi warna tinggi (kemungkinan bukan batu)"
        return False, "Objek terdeteksi sebagai potensi batu megalitikum"
    except Exception as e:
        return False, f"Error: {str(e)}"


# ==============================================
# FUNGSI CEK KUALITAS
# ==============================================
def cek_kualitas_gambar(image):
    try:
        arr = np.array(image.convert('L'))
        brightness = float(np.mean(arr))
        contrast = float(np.std(arr))
        blur_score = float(np.var(arr))
        kualitas = "Baik"
        pesan, rek = [], []
        if blur_score < 300:
            kualitas = "Buruk"
            pesan.append("• Gambar terlalu blur")
            rek.append("Gunakan gambar lebih tajam")
        elif blur_score < 600:
            kualitas = "Sedang"
            pesan.append("• Gambar sedikit blur")
            rek.append("Pastikan kamera tidak goyang")
        if brightness < 50:
            kualitas = "Sedang" if kualitas == "Baik" else kualitas
            pesan.append("• Terlalu gelap")
            rek.append("Tambahkan pencahayaan")
        elif brightness > 200:
            kualitas = "Sedang" if kualitas == "Baik" else kualitas
            pesan.append("• Terlalu terang")
            rek.append("Kurangi pencahayaan")
        if contrast < 30:
            kualitas = "Sedang" if kualitas == "Baik" else kualitas
            pesan.append("• Kontras rendah")
            rek.append("Pilih gambar lebih kontras")
        return kualitas, "\n".join(pesan) if pesan else "Kualitas gambar baik ✅", "\n".join(rek), blur_score, brightness, contrast
    except Exception as e:
        return "Tidak diketahui", f"Error: {str(e)}", "", 0, 128, 50


# ==============================================
# FUNGSI LOAD CLASS NAMES & MODEL INFO
# ==============================================
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        return ["Arca", "dolmen", "menhir", "dakon", "batu_datar", "Kubur_batu", "Lesung_batu"]


@st.cache_data
def load_model_info():
    try:
        with open('model_info.json', 'r') as f:
            return json.load(f)
    except:
        return {'test_accuracy': 0.9735, 'best_val_accuracy_phase2': 0.9940,
                'test_loss': 0.7564, 'test_auc': 0.9990}


# ==============================================
# FUNGSI PREDIKSI
# ==============================================
def predict_tflite(interp, input_details, output_details, image):
    img = image.convert('RGB').resize((224, 224))
    inp = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    interp.set_tensor(input_details[0]['index'], inp)
    interp.invoke()
    return interp.get_tensor(output_details[0]['index'])[0]


# ==============================================
# FUNGSI BUAT PDF
# ==============================================
def buat_pdf_hasil(nama_file, kelas, confidence, top3, deskripsi, kualitas="", warning=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(102, 126, 234)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_font("Arial", size=15, style='B')
    pdf.set_text_color(255, 255, 255)
    pdf.ln(8)
    pdf.cell(200, 10, txt="LAPORAN KLASIFIKASI BATU MEGALITIKUM", ln=1, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 8, txt="Informasi Prediksi", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(200, 7, txt=f"  File Gambar     : {nama_file}", ln=1, fill=True)
    pdf.cell(200, 7, txt=f"  Hasil Prediksi  : {kelas}", ln=1)
    pdf.cell(200, 7, txt=f"  Confidence      : {confidence:.2%}", ln=1, fill=True)
    pdf.cell(200, 7, txt=f"  Kualitas Gambar : {kualitas}", ln=1)
    if warning:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(200, 7, txt=f"  Catatan         : {warning}", ln=1)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 8, txt="Top 3 Prediksi:", ln=1)
    pdf.set_font("Arial", size=11)
    for i, (k, c) in enumerate(top3):
        pdf.set_fill_color(220, 255, 220) if i == 0 else pdf.set_fill_color(245, 245, 245)
        pdf.cell(200, 7, txt=f"  {i+1}. {k:<20} {c:.2%}", ln=1, fill=True)
    pdf.ln(4)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 8, txt="Deskripsi Kelas:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=deskripsi)
    pdf.ln(3)
    pdf.set_font("Arial", size=9, style='I')
    pdf.set_text_color(150, 150, 150)
    pdf.cell(200, 8, txt="Sistem Klasifikasi Batu Megalitikum Pagar Alam - ResNet50 Transfer Learning", ln=1, align='C')
    return pdf.output(dest='S').encode('latin1')


# ==============================================
# INISIALISASI
# ==============================================
interpreter, input_details, output_details = download_and_load_model()
class_names = load_class_names()
model_info = load_model_info()

# ==============================================
# SIDEBAR
# ==============================================
with st.sidebar:
    st.markdown("## 🗿 Megalitikum")
    st.markdown("**Sistem Klasifikasi Batu**")
    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    if interpreter is not None:
        ca, cb = st.columns(2)
        with ca:
            st.metric("Test Acc", f"{model_info.get('test_accuracy', 0.9735)*100:.2f}%")
        with cb:
            st.metric("Best Val", f"{model_info.get('best_val_accuracy_phase2', 0.9940)*100:.2f}%")
        st.caption(f"AUC: {model_info.get('test_auc', 0.9990):.4f}")
    else:
        st.error("⚠️ Model belum tersedia")
    st.markdown("---")
    st.markdown("### 🗿 Info Kelas")
    for nama, desk in DESKRIPSI_KELAS.items():
        with st.expander(f"🪨 {nama}"):
            st.write(desk)
    st.markdown("---")
    st.caption("© 2025 - Klasifikasi Megalitikum Pagar Alam")

# ==============================================
# CEK MODEL
# ==============================================
if interpreter is None:
    st.error("""
    ❌ **Model tidak dapat dimuat!**

    Pastikan:
    1. FILE_ID sudah benar di `app.py` (baris ~40)
    2. File `.tflite` sudah di-share **"Anyone with the link"**
    3. Cek `requirements.txt` sudah menggunakan versi yang kompatibel
    """)
    st.stop()

# ==============================================
# HEADER
# ==============================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 1.5rem;">
    <h1 style="color: white; margin: 0; font-size: 2rem;">🗿 Klasifikasi Batu Megalitikum</h1>
    <p style="color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0;">
        Pagar Alam, Sumatera Selatan | ResNet50 Transfer Learning
    </p>
    <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0.3rem 0 0 0;">
        Akurasi: 97.35% | 7 Kelas Megalitikum
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================
# TAB
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Prediksi", "ℹ️ Info Model", "📖 Panduan", "🔍 Filter Konten"
])

# ══════════════════════════════════════════════
# TAB 1: PREDIKSI
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 📤 Pilih Gambar Batu Megalitikum")

    sumber = st.radio("Sumber gambar:",
                      ["📁 Upload dari File", "📷 Ambil dengan Kamera"],
                      horizontal=True)

    gambar = None
    if sumber == "📁 Upload dari File":
        gambar = st.file_uploader("Pilih file gambar (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    else:
        st.info("⚠️ Browser akan meminta izin akses kamera. Klik **Allow** untuk melanjutkan.")
        gambar = st.camera_input("Ambil foto batu megalitikum")

    if gambar:
        image = Image.open(gambar)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Gambar Asli", use_container_width=True)

        kualitas, pesan_kualitas, rekomendasi, blur_score, brightness, contrast = cek_kualitas_gambar(image)
        is_non_megalith, deteksi_msg = detect_non_megalith(image)

        with col2:
            if kualitas == "Baik":
                st.success(f"✅ Kualitas Gambar: **{kualitas}**")
            elif kualitas == "Sedang":
                st.warning(f"⚠️ Kualitas Gambar: **{kualitas}**")
                st.info(f"Masalah:\n{pesan_kualitas}")
                st.caption(f"💡 Saran: {rekomendasi}")
            else:
                st.error(f"❌ Kualitas Gambar: **{kualitas}**")
                st.info(f"Masalah:\n{pesan_kualitas}")
                st.caption(f"💡 Saran: {rekomendasi}")

        if is_non_megalith:
            st.error(f"❌ Deteksi Konten: {deteksi_msg}")
            st.warning("⚠️ Model dirancang khusus untuk batu megalitikum.")
            lanjut = st.checkbox("✅ Saya mengerti, tetap lanjutkan prediksi")
            if not lanjut:
                st.stop()

        st.markdown("---")

        if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
            with st.spinner("🔍 Menganalisis gambar..."):
                enhanced_image = adaptive_enhancement(image, brightness, contrast)
                with col2:
                    st.image(enhanced_image, caption="✨ Setelah Enhancement", use_container_width=True)

                predictions = predict_tflite(interpreter, input_details, output_details, enhanced_image)
                pred_idx = int(np.argmax(predictions))
                pred_class = class_names[pred_idx]
                confidence = float(predictions[pred_idx])
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3 = [(class_names[i], float(predictions[i])) for i in top_3_idx]

                st.markdown("---")
                st.markdown("## 🎯 Hasil Prediksi")

                conf_color = "#27ae60" if confidence >= 0.8 else "#f39c12" if confidence >= 0.6 else "#e74c3c"
                conf_label = "Tinggi ✅" if confidence >= 0.8 else "Sedang ⚠️" if confidence >= 0.6 else "Rendah ❌"

                warning_msg = ""
                if is_non_megalith:
                    warning_msg = "Gambar terdeteksi non-batu — prediksi TIDAK AKURAT"
                    st.error(f"⚠️ {warning_msg}")
                elif confidence < 0.6:
                    warning_msg = "Confidence rendah — prediksi mungkin kurang akurat"
                    st.warning(f"⚠️ {warning_msg}")

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#f8f9ff 0%,#e8f0ff 100%);
                            padding:1.5rem; border-radius:12px;
                            border-left:5px solid #667eea;
                            box-shadow:0 2px 10px rgba(102,126,234,0.15);
                            margin-bottom:1rem;">
                    <h3 style="color:#667eea; margin:0 0 0.5rem 0;">🏆 {pred_class}</h3>
                    <p style="margin:0; font-size:1.1rem;">
                        Confidence: <strong style="color:{conf_color};">{confidence:.2%}</strong>
                        <span style="background:{conf_color}; color:white; padding:2px 8px;
                              border-radius:10px; font-size:0.8rem; margin-left:8px;">{conf_label}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                deskripsi = DESKRIPSI_KELAS.get(pred_class, "Tidak ada deskripsi.")
                st.info(f"📖 **{pred_class}**: {deskripsi}")

                st.markdown("---")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown("#### 🏆 Top 3 Prediksi")
                    medals = ["🥇", "🥈", "🥉"]
                    for i, (cls, conf) in enumerate(top_3):
                        bw = int(conf * 100)
                        color = "#667eea" if i == 0 else "#95a5a6"
                        st.markdown(f"""
                        <div style="margin-bottom:8px;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                                <span>{medals[i]} <strong>{cls}</strong></span>
                                <span style="color:{color};">{conf:.2%}</span>
                            </div>
                            <div style="background:#ecf0f1;border-radius:5px;height:8px;">
                                <div style="width:{bw}%;background:{color};border-radius:5px;height:8px;"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                with col_r2:
                    st.markdown("#### 📊 Probabilitas Semua Kelas")
                    sorted_idx = np.argsort(predictions)[::-1]
                    st.bar_chart({
                        "Kelas": [class_names[i] for i in sorted_idx],
                        "Probabilitas": [float(predictions[i]) for i in sorted_idx]
                    }, x="Kelas", y="Probabilitas", height=250)

                st.markdown("---")
                pdf_bytes = buat_pdf_hasil(
                    gambar.name if hasattr(gambar, 'name') else "foto_kamera.jpg",
                    pred_class, confidence, top_3, deskripsi, kualitas, warning_msg
                )
                st.download_button(
                    label="📥 Download Laporan PDF",
                    data=pdf_bytes,
                    file_name=f"laporan_{pred_class}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="secondary"
                )

# ══════════════════════════════════════════════
# TAB 2: INFO MODEL
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### ℹ️ Detail Arsitektur Model")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("#### 🏗️ Spesifikasi")
        st.json({
            "Arsitektur": "ResNet50 + Transfer Learning",
            "Pre-trained": "ImageNet",
            "Input Size": "224 × 224 × 3",
            "Jumlah Kelas": 7,
            "Total Parameter": "24,805,511",
            "Format": "TFLite (float32)"
        })
    with ci2:
        st.markdown("#### 📊 Hasil Evaluasi")
        st.json({
            "Test Accuracy": "97.35%",
            "Best Val Accuracy": "99.40% (Phase 2)",
            "Test Loss": "0.7564",
            "AUC": "0.9990",
            "Data Uji": "415 sampel",
            "Benar": "404 (97.35%)",
            "Salah": "11 (2.65%)"
        })

    st.markdown("---")
    st.markdown("#### 🔬 Fase Pelatihan")
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        st.info("**Fase 1 — Feature Extraction**\n- Epoch: 60\n- LR: 0.0001\n- Semua layer beku\n- Val Acc: **99.10%**")
    with cf2:
        st.success("**Fase 2 — Fine-Tuning ★**\n- Epoch: 50\n- LR: 0.00002\n- 100 layer beku\n- Val Acc: **99.40%**")
    with cf3:
        st.info("**Fase 3 — Gradual Unfreezing**\n- Epoch: 30\n- LR: 0.000005\n- 125 layer dilatih\n- Val Acc: **99.10%**")

    st.markdown("---")
    st.markdown("#### 📋 Performa per Kelas")
    import pandas as pd
    st.dataframe(pd.DataFrame({
        "Kelas": ["Arca","dolmen","menhir","dakon","batu_datar","Kubur_batu","Lesung_batu"],
        "Precision": [0.9825,1.0000,0.9833,0.9492,0.9833,0.9831,0.9344],
        "Recall":    [0.9492,1.0000,1.0000,0.9492,1.0000,0.9667,0.9500],
        "F1-Score":  [0.9655,1.0000,0.9916,0.9492,0.9916,0.9748,0.9421],
        "Support":   [59,59,59,59,59,60,60]
    }).set_index("Kelas"), use_container_width=True)

    st.markdown("---")
    st.warning("**⚠️ Keterbatasan:** Gambar blur/gelap/overexposed, resolusi sangat kecil, objek tertutup vegetasi, gambar NON-BATU, kemiripan visual Dakon & Lesung Batu.")

    st.markdown("---")
    st.markdown("#### 🗿 Deskripsi 7 Kelas")
    for nama, desk in DESKRIPSI_KELAS.items():
        with st.expander(f"🪨 {nama}"):
            st.write(desk)

# ══════════════════════════════════════════════
# TAB 3: PANDUAN
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📖 Panduan Penggunaan")
    st.markdown("""
    1. **Pilih sumber gambar** — Upload file atau foto dengan kamera
    2. **Pastikan kualitas baik** — tajam, cahaya cukup, objek jelas
    3. **Klik Prediksi Sekarang**
    4. **Lihat hasil** — kelas, confidence, top 3, grafik
    5. **Download laporan PDF**
    """)
    st.markdown("---")
    ct1, ct2 = st.columns(2)
    with ct1:
        st.success("✅ **Lakukan:** Foto jarak 50-100 cm, objek > 60% frame, cahaya merata, tampilkan ciri khas batu")
    with ct2:
        st.error("❌ **Hindari:** Foto terlalu jauh/dekat, latar ramai, cahaya langsung, gambar blur")
    st.info("📱 Dioptimalkan untuk smartphone. Gunakan kamera ponsel langsung di lapangan.")

# ══════════════════════════════════════════════
# TAB 4: FILTER KONTEN
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 🔍 Filter Konten Otomatis")
    st.markdown("""
    | Kondisi | Indikator | Hasil |
    |---------|-----------|-------|
    | Dominasi hijau | g > r×1.2 | Tumbuhan |
    | Dominasi biru | b > r×1.3 | Langit/air |
    | Variasi tinggi | variance > 500 | Non-batu |
    """)
    test_img = st.file_uploader("Upload gambar untuk diuji", type=['jpg','jpeg','png'], key="tf")
    if test_img:
        ti = Image.open(test_img)
        ct1, ct2 = st.columns(2)
        with ct1:
            st.image(ti, use_container_width=True)
        with ct2:
            is_non, msg = detect_non_megalith(ti)
            kual, pesan, _, blur, bright, cont = cek_kualitas_gambar(ti)
            st.error(f"❌ {msg}") if is_non else st.success(f"✅ {msg}")
            st.markdown(f"- Kualitas: **{kual}**\n- Blur: `{blur:.1f}`\n- Brightness: `{bright:.1f}`\n- Contrast: `{cont:.1f}`")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;padding:1rem 0;">
    <p style="margin:0;font-size:0.85rem;">
        🗿 <strong>Sistem Klasifikasi Batu Megalitikum Pagar Alam</strong><br>
        ResNet50 Transfer Learning | Akurasi 97.35% | 7 Kelas<br>
        <span style="font-size:0.75rem;">© 2025 — Penelitian Skripsi</span>
    </p>
</div>
""", unsafe_allow_html=True)
