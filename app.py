import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, r2_score

# Set page config
st.set_page_config(
    page_title="Analisis Model Prediksi Harga",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('BigBasket Products.csv')
    return df

# Custom prediction function untuk Decision Tree
def predict_sale_price(model, market_prices):
    predictions = model.predict(market_prices)
    market_prices_array = market_prices['market_price'].values
    adjusted_predictions = np.maximum(predictions, market_prices_array * 1.1)
    return adjusted_predictions

def perform_clustering(df):
    # Menggunakan StandardScaler untuk menstandarisasi data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['market_price', 'sale_price']])
    
    # Menggunakan KMeans untuk clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Menghitung silhouette score
    silhouette_avg = silhouette_score(scaled_data, df['cluster'])
    return df, silhouette_avg

def main():
    st.title("Dashboard Analisis Model Prediksi Harga")
    
    # Load data
    try:
        df = load_data()
    except:
        st.error("Error: File 'BigBasket Products.csv' tidak ditemukan")
        return

    # Lakukan clustering dan tambahkan kolom cluster ke df
    df, silhouette_avg = perform_clustering(df)

    # Sidebar untuk upload model
    st.sidebar.title("Upload Model")
    uploaded_pickle = st.sidebar.file_uploader("Upload file model (.pkl)", type=['pkl'])

    if uploaded_pickle is not None:
        try:
            # Load model
            model = pickle.load(uploaded_pickle)
            st.success("Model berhasil dimuat!")
        except Exception as e:
            st.error(f"Error saat memuat model: {e}")
            return

        # Tampilkan tabs untuk berbagai analisis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Info", "Feature Analysis", "Prediksi", "Visualisasi Model", "Analisis Data"])
        
        with tab1:
            st.header("Informasi Model")
            
            if isinstance(model, DecisionTreeRegressor):
                st.write("Tipe Model: Decision Tree Regressor")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kedalaman Pohon", model.get_depth())
                with col2:
                    st.metric("Jumlah Leaf Nodes", model.get_n_leaves())
                
                # Feature importance
                st.subheader("Feature Importance")
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': ['market_price'],
                    'Importance': importance
                })
                st.dataframe(importance_df)
            
            elif isinstance(model, LinearRegression):
                st.write("Tipe Model: Linear Regression")
                st.write("Koefisien:", model.coef_[0])
                st.write("Intercept:", model.intercept_)
            
            elif isinstance(model, KMeans):
                st.write("Tipe Model: KMeans Clustering")
                st.write("Jumlah Cluster:", model.n_clusters)
                st.write("Silhouette Score:", silhouette_avg)
        
        with tab2:
            st.header("Analisis Feature")
            
            # Scatter plot harga pasar vs harga jual
            st.subheader("Hubungan Harga Pasar vs Harga Jual")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='market_price', y='sale_price', alpha=0.5)
            plt.title("Scatter Plot Harga Pasar vs Harga Jual")
            plt.xlabel("Harga Pasar")
            plt.ylabel("Harga Jual")
            st.pyplot(fig)
            
            # Distribusi margin
            st.subheader("Analisis Margin")
            df['margin'] = ((df['sale_price'] - df['market_price']) / df['market_price']) * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='margin', bins=50)
            plt.title("Distribusi Margin Harga")
            plt.xlabel("Margin (%)")
            plt.ylabel("Frekuensi")
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rata-rata Margin", f"{df['margin'].mean():.2f}%")
            with col2:
                st.metric("Median Margin", f"{df['margin'].median():.2f}%")
        
        with tab3:
            st.header("Prediksi Harga")
            
            # Input untuk prediksi
            col1, col2 = st.columns(2)
            with col1:
                input_price = st.number_input("Masukkan Harga Pasar:", min_value=0.0, value=100.0)
            
            if st.button("Prediksi"):
                try:
                    if isinstance(model, LinearRegression):
                        prediction = model.predict([[input_price]])[0]
                        margin = ((prediction - input_price) / input_price) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediksi Harga Jual", f"${prediction:.2f}")
                        with col2:
                            st.metric("Margin Prediksi", f"{margin:.2f}%")
                        
                    elif isinstance(model, DecisionTreeRegressor):
                        prediction = predict_sale_price(model, pd.DataFrame({'market_price': [input_price]}))[0]
                        margin = ((prediction - input_price) / input_price) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediksi Harga Jual", f"${prediction:.2f}")
                        with col2:
                            st.metric("Margin Prediksi", f"{margin:.2f}%")
                except Exception as e:
                    st.error(f"Error saat melakukan prediksi: {e}")
                
                # Tampilkan contoh prediksi untuk beberapa harga
                st.subheader("Contoh Prediksi untuk Berbagai Harga")
                sample_prices = pd.DataFrame({'market_price': [100, 500, 1000, 2000, 5000]})
                predictions = predict_sale_price(model, sample_prices)
                
                results = []
                for market_price, pred in zip(sample_prices['market_price'], predictions):
                    margin = ((pred - market_price) / market_price) * 100
                    results.append({
                        'Harga Pasar': f"${market_price:.2f}",
                        'Prediksi Harga Jual': f"${pred:.2f}",
                        'Margin': f"{margin:.1f}%"
                    })
                st.table(pd.DataFrame(results))
        
        with tab4:
            st.header("Visualisasi Model")
            
            # Tambahkan log untuk memeriksa tipe model
            st.write(f"Tipe model yang dimuat: {type(model)}")
            
            if isinstance(model, DecisionTreeRegressor):
                st.subheader("Visualisasi Pohon Keputusan")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, feature_names=['market_price'], 
                         filled=True, rounded=True, fontsize=10)
                plt.title('Visualisasi Pohon Keputusan')
                st.pyplot(fig)
                
                # Plot prediksi vs aktual
                st.subheader("Prediksi vs Aktual")
                X = df[['market_price']]
                y_pred = model.predict(X)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.scatter(df['market_price'], df['sale_price'], alpha=0.5, label='Aktual')
                plt.scatter(df['market_price'], y_pred, alpha=0.5, label='Prediksi')
                plt.legend()
                plt.title('Perbandingan Harga Aktual vs Prediksi')
                plt.xlabel('Harga Pasar')
                plt.ylabel('Harga Jual')
                st.pyplot(fig)

            elif isinstance(model, KMeans):
                st.subheader("Visualisasi Hasil Clustering KMeans")
                
                # Pastikan df memiliki kolom 'cluster'
                if 'cluster' in df.columns:
                    # Plot hasil clustering
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x='market_price', y='sale_price', hue='cluster', palette='viridis', alpha=0.5)
                    plt.title('Hasil Clustering KMeans')
                    plt.xlabel('Harga Pasar')
                    plt.ylabel('Harga Jual')
                    st.pyplot(fig)
                else:
                    st.warning("Kolom 'cluster' tidak ditemukan dalam DataFrame.")

                # Visualisasi Elbow Method
                st.subheader("Visualisasi Elbow Method")
                inertias = []
                silhouette_scores = []
                k_range = range(2, 11)

                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(df[['market_price', 'sale_price']])  # Pastikan menggunakan dua fitur
                    inertias.append(kmeans.inertia_)
                    score = silhouette_score(df[['market_price', 'sale_price']], kmeans.labels_)
                    silhouette_scores.append(score)

                # Plot Elbow Method
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(k_range, inertias, 'bx-')
                plt.xlabel('Jumlah Cluster (k)')
                plt.ylabel('Inertia')
                plt.title('Elbow Method untuk Optimal k')
                st.pyplot(fig)

                # Plot Silhouette Score
                st.subheader("Visualisasi Silhouette Score")
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(k_range, silhouette_scores, 'rx-')
                plt.xlabel('Jumlah Cluster (k)')
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score untuk Optimal k')
                st.pyplot(fig)

            elif isinstance(model, LinearRegression):
                st.subheader("Koefisien Linear Regression")
                st.write("Koefisien:", model.coef_[0])
                st.write("Intercept:", model.intercept_)
                
                # Plot prediksi vs aktual untuk Linear Regression
                st.subheader("Prediksi vs Aktual")
                X = df[['market_price']]
                y_pred = model.predict(X)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.scatter(df['market_price'], df['sale_price'], alpha=0.5, label='Aktual')
                plt.scatter(df['market_price'], y_pred, alpha=0.5, label='Prediksi', color='orange')
                plt.plot(df['market_price'], y_pred, color='red')  # Garis regresi
                plt.legend()
                plt.title('Perbandingan Harga Aktual vs Prediksi')
                plt.xlabel('Harga Pasar')
                plt.ylabel('Harga Jual')
                
                # Hitung R²
                r2 = r2_score(df['sale_price'], y_pred)  # Hitung R²
                equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
                r2_text = f'R² = {r2:.4f}'
                plt.text(0.05, 0.95, equation + '\n' + r2_text, 
                         transform=plt.gca().transAxes, 
                         bbox=dict(facecolor='white', alpha=0.8))

                st.pyplot(fig)

            else:
                st.warning("Model yang dimuat tidak didukung untuk visualisasi.")
        
        with tab5:
            st.header("Analisis Data")
            st.markdown(""" 
            **Business Understanding**  
            Sebuah toko online dengan produk kebutuhan rumah yang terdapat beberapa barang, dari data tersebut kami ingin menganalisis data produk untuk memahami performa produk.
            """)
            st.markdown("""
            | Nama Kolom               | Deskripsi                                     |
            |--------------------------|-----------------------------------------------|
            | index                    | Indeks dari produk (sebagai penanda urutan).  |
            | product                  | Judul produk (seperti yang tercantum).        |
            | category                 | Kategori tempat produk diklasifikasikan.      |
            | sub_category             | Subkategori tempat produk dikelompokkan.      |
            | brand                    | Merek dari produk.                            |
            | sales_price              | Harga produk yang dijual di situs.            |
            | market_price             | Harga pasar dari produk.                      |
            | type                     | Jenis produk.                                 |
            | rating                   | Penilaian yang diberikan konsumen terhadap produk.|
            """)
        
    else:
        st.info("Silakan upload file model (.pkl) untuk melihat analisis")

if __name__ == "__main__":
    main()

