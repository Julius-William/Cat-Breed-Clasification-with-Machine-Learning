import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

def run():

    model = load_model('best_model.keras')

    st.title("Masukkan Gambar Yang Ingin Dipredict")

    uploaded_file = st.file_uploader("Masukkan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)
        

        pred = model.predict(img_arr)
        pred_class = np.argmax(pred, axis=1)[0]

        class_names = ['bengal', 'domestic_shorthair', 'maine_coon', 'ragdoll','siamese']

        st.subheader("🔍 Prediction Result")
        st.write("### Jenis Ras Kucing:", class_names[pred_class])

    else:
        st.info("Silakan upload gambar kendaraan terlebih dahulu.")

if __name__ == "__main__":
    run()