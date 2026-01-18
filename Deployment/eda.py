def run():  
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import os 
    import random
    from PIL import Image

    st.title('Klasifikasi Jenis/Ras Kucing dengan CNN')
    st.image('https://indomgb.s3.amazonaws.com/wp-content/uploads/2023/06/22012951/kucing-rumahan-unsplash.jpeg')

    st.markdown('''## Latar Belakang''')
    
    st.markdown('''Objektif utama dari Tugas ini adalah untuk membuat sebuah model dengan arsitektur ANN/ CNN. Arsitektur ANN dan CNN dapat digunakan untuk melakukan banyak hal, tapi yang akan dilakukan pada tugas ini adalah untuk mengklasifikasikan ras kucing berdasarkan gambar yang diberikan. Jadi hal yang pertama dilakukan adalah untuk membuat sebuah model dengan ANN dan melatihnya dengan gambar-gambar kucing dengan ras yang beda-beda. Tujuan dari model ini adalah untuk membantu pasar jual beli kucing, karena kucing adalah salah satu hewan dengan banyak ras yang berbeda dan mirip-mirip satu sama lain. Selain itu akan dilakukan berbaikan model ANN/CNN yang dibuat. Ada beberapa hal yang dapat digunakan untuk lah ini, salah satunya adalah transfer learning yang dimana menggunakan model yang sudah ada dan jauh lebih kompleks dan di implimentasikan ke model kita agar dapat meningkatkan hasil kita.''')
    
    st.markdown('''## Ciri-ciri Jenis/Ras Kucing''')
    kolom = st.selectbox(
        "Pilih kolom yang akan divisualisasikan.",
        options=["bengal",
    "domestic_shorthair",
    "maine_coon",
    "ragdoll",
    "siamese"]
    )

    labels = []
    labels.append(kolom)

    base_path = "Cat_Data"

    for i in labels:
        num_samples = 5
        fig, axes = plt.subplots(len(labels), num_samples, figsize=(15, 4))
    
        # Jika hanya 1 baris → jadikan axes 2D
        if len(labels) == 1:
            axes = axes.reshape(1, num_samples)
    
        for row, lbl in enumerate(labels):
            class_folder = os.path.join(base_path, lbl)
    
            img_files = [
                f for f in os.listdir(class_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
    
            sample_imgs = random.sample(img_files, min(num_samples, len(img_files)))
    
            for col, img_file in enumerate(sample_imgs):
                img_path = os.path.join(class_folder, img_file)
                img = Image.open(img_path)
    
                axes[row, col].imshow(img)
                axes[row, col].set_title(lbl if col == 0 else "")
                axes[row, col].axis('off')
    
        plt.tight_layout()
        st.pyplot(fig)

        if kolom == "bengal":
            st.markdown('Bengal: Kucing bengal adalah salah satu jenis kucing yang memiliki penampilan unik. Kucinng bengal memiliki pola yang membedangan dengan jenis kucing-kuncing lainnya. Mereka memiliki pola bulat-bulat hitam atau strip yang membuatnya mirip seperti macan, terutama macan tutul.')
        elif kolom == "domestic_shorthair":
            st.markdown('''Domestic Shorthair: Kucing domestic shorthair adalah kucing rumahan yang paling umum. Kucing ini memiliki bulu yang pendek dan memiliki banyak warna dan pola. Berikut adalah beberapa pola yang dapat dimiliki kucing domestic shorthair:
- Tuxedo
- Calico
- Tabby
- Orange
- Putih
- Hitam
- Dkk

Dapat dibilang domestic short hair adalah kucing yang tidak memiliki ras murni sehingga memiliki banyak pola danwaran yang berbeda.''')
        elif kolom == "maine_coon":
            st.markdown("Maine Coon: Kucing maine coon adalah salah satu ras kucing yang terbesar di dunia. Maine coon memiliki bulu yang cenderung panjang dam memiliki telinga yang tinggi dan besar dibandingkan kucing pada umumnya. Selain itu, bentuk kepala maine coon menjadi salah satu fitur yang membedakan dengan kucing lainnya yaitu muzzle kotak. Dapat dilihat bahwa dari hidung ke mulutnya membentuk kotak dan itu menjadi salah satu ciri khas ras maine coon. Lalu, dapat dikatakan bahwa maine coon memeiliki badan yang kuat dan berotot, sehingga memberi bentuk badan yang tidak seperti kucing pada umumnya.")
        elif kolom == "ragdoll":
            st.markdown("Ragdoll: Kucing ragdoll adalah ras kucing yang berukuran besar, memiiliki bulu panjang dan bulu yang sangat lembut. Beda dengan maine coon, ragdoll lebih membulat dan lebih lemas. Salah satu hal yang membua unik ras ragdoll adalah warna mata yang selalu biru. Hall-hal ini membuat pandampilan ragdoll lucu dan cantik.")
        else:
            st.markdown("Siamese: Kucing siamese memiliki muka yang denderung panjang dan seperti segitiga. Selain itu, siamese memiliki terling tinggi dan lebar di dasarnya. Siamese memiliki pola warna colorpoint yang berarti mukanya memiliki warna yang lebih gelap dobandingkan badannya. Dan salah satu fitur yang membadakan siamese adalah mata yang biru.")

    