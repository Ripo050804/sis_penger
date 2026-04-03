"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Fitur: Kamera, Upload, Info Kelas, Download PDF, Mobile-friendly
Dengan auto-download model dari Google Drive
Versi: Stabil untuk Streamlit Cloud
Kelas: Arca, dolmen, menhir, dakon, batu_datar, Kubur_batu, Lesung_batu
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import os
from fpdf import FPDF
from io import BytesIO
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
# DESKRIPSI KELAS (sudah diupdate: monolit → Lesung_batu)
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
# ⚠️  PENTING: CARA MENDAPATKAN FILE_ID YANG BENAR
# ==============================================
# 1. Buka Google Drive Anda
# 2. Klik kanan file "megalitikum_model.tflite"
# 3. Pilih "Get link" / "Share"
# 4. Pastikan akses = "Anyone with the link"
# 5. Copy link, contoh:
#    https://drive.google.com/file/d/1xAbCdEfGhIjKlMnOpQrStUvWxYz/view
# 6. FILE_ID = bagian setelah /d/ yaitu: 1xAbCdEfGhIjKlMnOpQrStUvWxYz
#
# ⚠️  185Lo6wOmi7zL47xyEZ060_FFmHF_9SwF adalah ID FOLDER, bukan ID FILE
#     Anda perlu ID file .tflite yang ada di dalam folder tersebut
# ==============================================
FILE_ID = "1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx"  # ← GANTI INI

# ==============================================
# FUNGSI DOWNLOAD MODEL DARI GOOGLE DRIVE
# ==============================================
@st.cache_resource
def download_and_load_model():
    """Download model TFLite dari Google Drive jika belum ada"""

    model_path = "megalitikum_model.tflite"

    # Validasi FILE_ID
    if FILE_ID == "GANTI_DENGAN_FILE_ID_TFLITE_ANDA":
        st.error("""
        ❌ **FILE_ID belum diisi!**

        Cara mendapatkan FILE_ID:
        1. Buka Google Drive → folder megalitikum_final123
        2. Klik kanan file **megalitikum_model.tflite** → Get link
        3. Pastikan akses = "Anyone with the link"
        4. Copy link: `https://drive.google.com/file/d/**FILE_ID**/view`
        5. Ganti variable FILE_ID di baris ~50 kode ini
        """)
        return None, None, None

    # Cek apakah model sudah ada
    if not os.path.exists(model_path):
        with st.spinner("🔄 Mendownload model (±96 MB) dari Google Drive... Harap tunggu."):
            try:
                # Metode 1: gdown (lebih andal untuk file besar)
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, model_path, quiet=False, fuzzy=True)
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                    st.success("✅ Model berhasil didownload!")
                else:
                    raise Exception("File terlalu kecil, kemungkinan gagal download")
            except Exception as e1:
                try:
                    # Metode 2: requests sebagai fallback
                    st.warning("⚠️ Mencoba metode alternatif...")
                    session = requests.Session()
                    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
                    response = session.get(url, stream=True, timeout=60)

                    # Handle konfirmasi virus scan Google
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={FILE_ID}"
                            response = session.get(url, stream=True, timeout=60)
                            break

                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)

                    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                        st.success("✅ Model berhasil didownload (metode alternatif)!")
                    else:
                        raise Exception("File hasil download tidak valid")
                except Exception as e2:
                    st.error(f"❌ Gagal download model: {str(e2)}")
                    st.info("💡 Pastikan FILE_ID benar dan file sudah di-share 'Anyone with the link'")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    return None, None, None

    # Load model TFLite
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None, None, None


# ==============================================
# FUNGSI ENHANCEMENT GAMBAR
# ==============================================
def enhance_image(image):
    """Tingkatkan kualitas gambar untuk prediksi lebih akurat"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    return image


def adaptive_enhancement(image, brightness, contrast):
    """Enhancement adaptif berdasarkan skor kualitas gambar"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.filter(ImageFilter.SHARPEN)
    if contrast < 40:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
    if brightness < 80:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)
    elif brightness > 180:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.8)
    return image


# ==============================================
# FUNGSI DETEKSI OBJEK NON-BATU
# ==============================================
def detect_non_megalith(image):
    """Deteksi sederhana apakah gambar mengandung objek non-batu"""
    try:
        img = image.convert('RGB')
        r, g, b = img.split()
        r_mean = np.mean(np.array(r))
        g_mean = np.mean(np.array(g))
        b_mean = np.mean(np.array(b))

        if g_mean > r_mean * 1.2 and g_mean > b_mean * 1.2:
            return True, "Gambar didominasi warna hijau (mungkin tumbuhan/vegetasi)"
        if b_mean > r_mean * 1.3 and b_mean > g_mean * 1.3:
            return True, "Gambar didominasi warna biru (mungkin langit atau air)"
        rgb_variance = np.var([r_mean, g_mean, b_mean])
        if rgb_variance > 500:
            return True, "Gambar memiliki variasi warna tinggi (kemungkinan bukan batu)"
        return False, "Objek terdeteksi sebagai potensi batu megalitikum"
    except Exception as e:
        return False, f"Error deteksi: {str(e)}"


# ==============================================
# FUNGSI DETEKSI KUALITAS GAMBAR
# ==============================================
def cek_kualitas_gambar(image):
    """Deteksi kualitas gambar menggunakan Pillow"""
    try:
        gray = image.convert('L')
        img_array = np.array(gray)
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        blur_score = np.var(img_array)

        kualitas = "Baik"
        pesan = []
        rekomendasi = []

        if blur_score < 300:
            kualitas = "Buruk"
            pesan.append("• Gambar terlalu blur/kabur")
            rekomendasi.append("Gunakan gambar yang lebih tajam")
        elif blur_score < 600:
            kualitas = "Sedang"
            pesan.append("• Gambar sedikit blur")
            rekomendasi.append("Pastikan kamera tidak goyang saat memotret")

        if brightness < 50:
            kualitas = "Sedang" if kualitas == "Baik" else kualitas
            pesan.append("• Gambar terlalu gelap")
            rekomendasi.append("Gunakan pencahayaan yang lebih terang")
        elif brightness > 200:
            kualitas = "Sedang" if kualitas == "Baik" else kualitas
            pesan.append("• Gambar terlalu terang/overexposed")
            rekomendasi.append("Kurangi pencahayaan atau hindari cahaya langsung")

        if contrast < 30:
            kualitas = "Sedang" if kualitas == "Baik" else kualitas
            pesan.append("• Kontras gambar rendah")
            rekomendasi.append("Pilih gambar dengan perbedaan warna yang lebih jelas")

        pesan_text = "\n".join(pesan) if pesan else "Kualitas gambar baik ✅"
        rekomendasi_text = "\n".join(rekomendasi) if rekomendasi else "Tidak perlu perbaikan"

        return kualitas, pesan_text, rekomendasi_text, blur_score, brightness, contrast
    except Exception as e:
        return "Tidak diketahui", f"Error: {str(e)}", "", 0, 128, 50


# ==============================================
# FUNGSI LOAD CLASS NAMES & MODEL INFO
# ==============================================
@st.cache_data
def load_class_names():
    """Load nama kelas dari file JSON atau gunakan default"""
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        # Default kelas yang sudah diupdate (monolit → Lesung_batu)
        return ["Arca", "dolmen", "menhir", "dakon", "batu_datar", "Kubur_batu", "Lesung_batu"]


@st.cache_data
def load_model_info():
    """Load informasi performa model"""
    try:
        with open('model_info.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'test_accuracy': 0.9735,
            'best_val_accuracy_phase2': 0.9940,
            'test_loss': 0.7564,
            'test_auc': 0.9990
        }


# ==============================================
# FUNGSI PREDIKSI TFLITE
# ==============================================
def predict_tflite(interpreter, input_details, output_details, image):
    """Jalankan prediksi menggunakan model TFLite"""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


# ==============================================
# FUNGSI BUAT PDF LAPORAN HASIL
# ==============================================
def buat_pdf_hasil(nama_file, kelas, confidence, top3, deskripsi, kualitas="", warning=""):
    """Buat laporan PDF hasil prediksi"""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(102, 126, 234)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_font("Arial", size=16, style='B')
    pdf.set_text_color(255, 255, 255)
    pdf.cell(200, 15, txt="", ln=1)
    pdf.cell(200, 10, txt="LAPORAN KLASIFIKASI BATU MEGALITIKUM", ln=1, align='C')
    pdf.ln(5)

    # Reset warna
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=11)

    # Info file
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Informasi Prediksi", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(200, 8, txt=f"  File Gambar    : {nama_file}", ln=1, fill=True)
    pdf.cell(200, 8, txt=f"  Hasil Prediksi : {kelas}", ln=1)
    pdf.cell(200, 8, txt=f"  Confidence     : {confidence:.2%}", ln=1, fill=True)
    pdf.cell(200, 8, txt=f"  Kualitas Gambar: {kualitas}", ln=1)
    if warning:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(200, 8, txt=f"  Catatan        : {warning}", ln=1)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Top 3 Prediksi
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Top 3 Prediksi:", ln=1)
    pdf.set_font("Arial", size=11)
    medals = ["1.", "2.", "3."]
    for i, (k, c) in enumerate(top3):
        fill = True if i == 0 else False
        pdf.set_fill_color(220, 255, 220) if i == 0 else pdf.set_fill_color(240, 240, 240)
        pdf.cell(200, 8, txt=f"  {medals[i]} {k:<20} {c:.2%}", ln=1, fill=fill)
    pdf.ln(5)

    # Deskripsi
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Deskripsi Kelas:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=deskripsi)
    pdf.ln(5)

    # Footer
    pdf.set_font("Arial", size=9, style='I')
    pdf.set_text_color(150, 150, 150)
    pdf.cell(200, 10, txt="Sistem Klasifikasi Batu Megalitikum Pagar Alam - ResNet50 Transfer Learning", ln=1, align='C')

    return pdf.output(dest='S').encode('latin1')


# ==============================================
# LOAD SEMUA DATA DAN MODEL
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
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Test Acc", f"{model_info.get('test_accuracy', 0.9735)*100:.2f}%")
        with col_b:
            st.metric("Best Val", f"{model_info.get('best_val_accuracy_phase2', 0.9940)*100:.2f}%")
        st.caption(f"AUC: {model_info.get('test_auc', 0.9990):.4f} | Loss: {model_info.get('test_loss', 0.7564):.4f}")
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
# CEK MODEL - STOP JIKA TIDAK TERSEDIA
# ==============================================
if interpreter is None:
    st.error("""
    ❌ **Model tidak dapat dimuat!**

    **Langkah yang harus dilakukan:**

    1. Buka Google Drive Anda
    2. Masuk ke folder **megalitikum_final123**
    3. Klik kanan file **megalitikum_model.tflite** → **Get link**
    4. Pastikan akses = **"Anyone with the link"**
    5. Copy link: `https://drive.google.com/file/d/`**`FILE_ID_INI`**`/view`
    6. Ganti variable `FILE_ID` di baris ~50 pada kode `app.py`
    7. Push ulang ke GitHub → Streamlit akan otomatis reload
    """)
    st.stop()

# ==============================================
# HEADER UTAMA
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
# TAB UTAMA
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Prediksi",
    "ℹ️ Info Model",
    "📖 Panduan",
    "🔍 Filter Konten"
])

# ══════════════════════════════════════════════
# TAB 1: PREDIKSI
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 📤 Pilih Gambar Batu Megalitikum")

    sumber = st.radio(
        "Sumber gambar:",
        ["📁 Upload dari File", "📷 Ambil dengan Kamera"],
        horizontal=True
    )

    gambar = None
    if sumber == "📁 Upload dari File":
        gambar = st.file_uploader(
            "Pilih file gambar (JPG/PNG)...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar batu megalitikum untuk diklasifikasikan"
        )
    else:
        st.info("⚠️ Browser akan meminta izin akses kamera. Klik **Allow** untuk melanjutkan.")
        gambar = st.camera_input("Ambil foto batu megalitikum")

    if gambar:
        image = Image.open(gambar)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Gambar Asli", use_container_width=True)

        # Cek kualitas gambar
        kualitas, pesan_kualitas, rekomendasi, blur_score, brightness, contrast = cek_kualitas_gambar(image)

        # Deteksi objek non-batu
        is_non_megalith, deteksi_msg = detect_non_megalith(image)

        # Tampilkan status kualitas
        with col2:
            if kualitas == "Baik":
                st.success(f"✅ **Kualitas Gambar: {kualitas}**")
            elif kualitas == "Sedang":
                st.warning(f"⚠️ **Kualitas Gambar: {kualitas}**")
                st.info(f"**Masalah terdeteksi:**\n{pesan_kualitas}")
                st.caption(f"💡 **Saran:** {rekomendasi}")
            else:
                st.error(f"❌ **Kualitas Gambar: {kualitas}**")
                st.info(f"**Masalah:**\n{pesan_kualitas}")
                st.caption(f"💡 **Saran:** {rekomendasi}")

        # Peringatan objek non-batu
        if is_non_megalith:
            st.error(f"❌ **Deteksi Konten:** {deteksi_msg}")
            st.warning("⚠️ Model ini dirancang khusus untuk gambar batu megalitikum. Hasil prediksi pada gambar non-batu tidak akan akurat.")
            lanjut = st.checkbox("✅ Saya mengerti, tetap lanjutkan prediksi")
            if not lanjut:
                st.stop()

        st.markdown("---")

        # Tombol Prediksi
        if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
            with st.spinner("🔍 Menganalisis gambar..."):

                # Proses enhancement
                enhanced_image = adaptive_enhancement(image, brightness, contrast)

                with col2:
                    st.image(enhanced_image, caption="✨ Gambar setelah Enhancement", use_container_width=True)

                # Jalankan prediksi
                predictions = predict_tflite(interpreter, input_details, output_details, enhanced_image)
                pred_idx = int(np.argmax(predictions))
                pred_class = class_names[pred_idx]
                confidence = float(predictions[pred_idx])

                # Top 3 prediksi
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3 = [(class_names[i], float(predictions[i])) for i in top_3_idx]

                st.markdown("---")
                st.markdown("## 🎯 Hasil Prediksi")

                # Tentukan warna confidence
                if confidence >= 0.8:
                    conf_color = "#27ae60"
                    conf_label = "Tinggi"
                elif confidence >= 0.6:
                    conf_color = "#f39c12"
                    conf_label = "Sedang"
                else:
                    conf_color = "#e74c3c"
                    conf_label = "Rendah"

                # Warning confidence rendah
                warning_msg = ""
                if is_non_megalith:
                    warning_msg = "Gambar terdeteksi non-batu — prediksi TIDAK AKURAT"
                    st.error(f"⚠️ {warning_msg}")
                elif confidence < 0.6:
                    warning_msg = "Confidence rendah — prediksi mungkin kurang akurat"
                    st.warning(f"⚠️ {warning_msg}")

                # Kotak hasil utama
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
                            padding: 1.5rem; border-radius: 12px;
                            border-left: 5px solid #667eea;
                            box-shadow: 0 2px 10px rgba(102,126,234,0.15);
                            margin-bottom: 1rem;">
                    <h3 style="color: #667eea; margin: 0 0 0.5rem 0;">🏆 {pred_class}</h3>
                    <p style="margin: 0; font-size: 1.1rem;">
                        Confidence: <strong style="color: {conf_color};">{confidence:.2%}</strong>
                        <span style="background: {conf_color}; color: white; padding: 2px 8px;
                              border-radius: 10px; font-size: 0.8rem; margin-left: 8px;">
                            {conf_label}
                        </span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Deskripsi kelas
                deskripsi = DESKRIPSI_KELAS.get(pred_class, "Tidak ada deskripsi tersedia.")
                st.info(f"📖 **{pred_class}**: {deskripsi}")

                st.markdown("---")

                # Top 3 prediksi
                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    st.markdown("#### 🏆 Top 3 Prediksi")
                    medals = ["🥇", "🥈", "🥉"]
                    for i, (cls, conf) in enumerate(top_3):
                        bar_width = int(conf * 100)
                        color = "#667eea" if i == 0 else "#95a5a6"
                        st.markdown(f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                <span>{medals[i]} <strong>{cls}</strong></span>
                                <span style="color: {color};">{conf:.2%}</span>
                            </div>
                            <div style="background: #ecf0f1; border-radius: 5px; height: 8px;">
                                <div style="width: {bar_width}%; background: {color};
                                     border-radius: 5px; height: 8px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_r2:
                    st.markdown("#### 📊 Distribusi Probabilitas")
                    sorted_idx = np.argsort(predictions)[::-1]
                    chart_data = {
                        "Kelas": [class_names[i] for i in sorted_idx],
                        "Probabilitas": [float(predictions[i]) for i in sorted_idx]
                    }
                    st.bar_chart(chart_data, x="Kelas", y="Probabilitas", height=250)

                st.markdown("---")

                # Download PDF
                pdf_bytes = buat_pdf_hasil(
                    gambar.name if hasattr(gambar, 'name') else "foto_kamera.jpg",
                    pred_class,
                    confidence,
                    top_3,
                    deskripsi,
                    kualitas,
                    warning_msg
                )
                st.download_button(
                    label="📥 Download Laporan PDF",
                    data=pdf_bytes,
                    file_name=f"laporan_klasifikasi_{pred_class}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="secondary"
                )

# ══════════════════════════════════════════════
# TAB 2: INFO MODEL
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### ℹ️ Detail Arsitektur Model")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("#### 🏗️ Spesifikasi Model")
        st.json({
            "Arsitektur": "ResNet50 + Transfer Learning",
            "Dataset": "ImageNet (pre-trained)",
            "Input Size": "224 × 224 × 3 (RGB)",
            "Jumlah Kelas": 7,
            "Total Parameter": "24,805,511",
            "Trainable (Head)": "1,216,007",
            "Format Model": "TFLite (float32)"
        })

    with col_i2:
        st.markdown("#### 📊 Hasil Evaluasi")
        st.json({
            "Test Accuracy": f"{model_info.get('test_accuracy', 0.9735):.2%}",
            "Best Val Accuracy (Phase 2)": f"{model_info.get('best_val_accuracy_phase2', 0.9940):.2%}",
            "Test Loss": f"{model_info.get('test_loss', 0.7564):.4f}",
            "AUC": f"{model_info.get('test_auc', 0.9990):.4f}",
            "Data Uji": "415 sampel",
            "Prediksi Benar": "404 sampel (97.35%)",
            "Prediksi Salah": "11 sampel (2.65%)"
        })

    st.markdown("---")
    st.markdown("#### 🔬 Fase Pelatihan")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        **Fase 1 — Feature Extraction**
        - Epoch: 60
        - LR: 0.0001
        - Layer beku: semua (175)
        - Val Acc: **99.10%**
        """)
    with col_f2:
        st.markdown("""
        **Fase 2 — Fine-Tuning ★**
        - Epoch: 50
        - LR: 0.00002
        - Layer beku: 100 pertama
        - Val Acc: **99.40%**
        """)
    with col_f3:
        st.markdown("""
        **Fase 3 — Gradual Unfreezing**
        - Epoch: 30
        - LR: 0.000005
        - Layer dilatih: 125
        - Val Acc: **99.10%**
        """)

    st.markdown("---")
    st.markdown("#### 📋 Performa per Kelas")
    perf_data = {
        "Kelas": ["Arca", "dolmen", "menhir", "dakon", "batu_datar", "Kubur_batu", "Lesung_batu"],
        "Precision": [0.9825, 1.0000, 0.9833, 0.9492, 0.9833, 0.9831, 0.9344],
        "Recall": [0.9492, 1.0000, 1.0000, 0.9492, 1.0000, 0.9667, 0.9500],
        "F1-Score": [0.9655, 1.0000, 0.9916, 0.9492, 0.9916, 0.9748, 0.9421],
    }
    import pandas as pd
    df = pd.DataFrame(perf_data)
    st.dataframe(df.set_index("Kelas"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚠️ Keterbatasan Model")
    st.warning("""
    **Model ini memiliki keterbatasan pada:**
    - Gambar dengan kualitas rendah (blur, gelap, overexposed)
    - Gambar resolusi sangat kecil (< 100×100 piksel)
    - Objek yang tidak terlihat jelas atau tertutup vegetasi
    - Pencahayaan ekstrem tanpa shadow detail
    - **Gambar NON-BATU** (hewan, tumbuhan, manusia, bangunan, dll)
    - Kemiripan visual antara kelas Dakon dan Lesung Batu
    """)

    st.markdown("---")
    st.markdown("#### 🗿 Deskripsi Lengkap 7 Kelas")
    for nama, desk in DESKRIPSI_KELAS.items():
        with st.expander(f"🪨 {nama}"):
            st.write(desk)


# ══════════════════════════════════════════════
# TAB 3: PANDUAN
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📖 Panduan Penggunaan Aplikasi")

    st.markdown("""
    #### 🚀 Langkah-langkah Prediksi

    **1. Pilih Sumber Gambar**
    - **Upload dari File**: Pilih gambar dari galeri/storage perangkat Anda
    - **Ambil dengan Kamera**: Foto langsung menggunakan kamera perangkat

    **2. Pastikan Kualitas Gambar Baik**
    - Objek batu terlihat jelas dan tidak blur
    - Pencahayaan cukup (tidak terlalu gelap/terang)
    - Fokus pada bagian utama batu megalitikum

    **3. Klik Tombol Prediksi**
    - Sistem akan menganalisis gambar secara otomatis
    - Proses membutuhkan beberapa detik

    **4. Lihat Hasil Prediksi**
    - Kelas dengan confidence tertinggi ditampilkan sebagai hasil utama
    - Top 3 prediksi beserta probabilitasnya juga ditampilkan
    - Grafik distribusi probabilitas untuk semua kelas

    **5. Download Laporan PDF**
    - Klik tombol **Download Laporan PDF**
    - Laporan memuat hasil prediksi, top 3, dan deskripsi kelas
    """)

    st.markdown("---")
    st.markdown("#### 💡 Tips untuk Hasil Prediksi Terbaik")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.success("""
        ✅ **Lakukan ini:**
        - Foto dari jarak 50-100 cm
        - Pastikan objek mengisi > 60% frame
        - Gunakan cahaya alami atau lampu merata
        - Foto dari sudut yang menampilkan ciri khas batu
        - Bersihkan lumut/kotoran yang menutupi batu
        """)
    with col_t2:
        st.error("""
        ❌ **Hindari ini:**
        - Foto terlalu jauh atau terlalu dekat
        - Latar belakang yang sangat ramai
        - Cahaya langsung matahari (bayangan keras)
        - Gambar yang blur atau goyang
        - Foto yang didominasi vegetasi/tanah
        """)

    st.markdown("---")
    st.info("""
    📱 **Optimasi Mobile**

    Aplikasi ini dioptimalkan untuk digunakan di smartphone. Anda dapat langsung
    menggunakan kamera ponsel untuk mengambil foto artefak megalitikum dan
    mendapatkan hasil klasifikasi secara real-time.
    """)


# ══════════════════════════════════════════════
# TAB 4: FILTER KONTEN
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 🔍 Filter Konten Otomatis")
    st.markdown("""
    Aplikasi ini dilengkapi dengan **sistem filter konten** untuk mendeteksi
    gambar yang bukan batu megalitikum sebelum diproses oleh model.

    #### ⚙️ Cara Kerja Filter
    | Kondisi | Indikator | Tindakan |
    |---------|-----------|----------|
    | Dominasi hijau | g_mean > r_mean × 1.2 | Deteksi tumbuhan |
    | Dominasi biru | b_mean > r_mean × 1.3 | Deteksi langit/air |
    | Variasi warna tinggi | variance > 500 | Kemungkinan non-batu |
    | Blur rendah | var < 300 | Kualitas buruk |
    | Terlalu gelap | brightness < 50 | Kualitas sedang |
    | Terlalu terang | brightness > 200 | Kualitas sedang |

    #### ⚠️ Penting
    Filter ini bersifat heuristik sederhana berbasis analisis warna.
    Beberapa gambar batu dengan lumut atau di luar ruangan mungkin
    memicu peringatan. Anda tetap dapat melanjutkan prediksi jika yakin
    gambar adalah batu megalitikum.
    """)

    st.markdown("---")
    st.markdown("#### 🧪 Uji Filter dengan Gambar")
    test_img = st.file_uploader(
        "Upload gambar untuk diuji filternya",
        type=['jpg', 'jpeg', 'png'],
        key="test_filter"
    )
    if test_img:
        test_image = Image.open(test_img)
        col_tf1, col_tf2 = st.columns(2)
        with col_tf1:
            st.image(test_image, caption="Gambar yang diuji", use_container_width=True)
        with col_tf2:
            is_non, msg = detect_non_megalith(test_image)
            kual, pesan, _, blur, bright, cont = cek_kualitas_gambar(test_image)
            if is_non:
                st.error(f"❌ **Filter Konten:** {msg}")
            else:
                st.success(f"✅ **Filter Konten:** {msg}")
            st.markdown(f"""
            **Detail Analisis:**
            - Kualitas: **{kual}**
            - Blur score: `{blur:.1f}`
            - Brightness: `{bright:.1f}`
            - Contrast: `{cont:.1f}`
            """)
            if pesan and pesan != "Kualitas gambar baik ✅":
                st.warning(f"⚠️ {pesan}")


# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem 0;">
    <p style="margin: 0; font-size: 0.85rem;">
        🗿 <strong>Sistem Klasifikasi Batu Megalitikum Pagar Alam</strong><br>
        ResNet50 Transfer Learning | Akurasi 97.35% | 7 Kelas<br>
        <span style="font-size: 0.75rem;">© 2025 — Penelitian Skripsi</span>
    </p>
</div>
""", unsafe_allow_html=True)
