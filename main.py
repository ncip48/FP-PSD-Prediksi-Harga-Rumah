import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_data, train_evaluate_regression, check_assumptions

def main():
    # Load data
    df = pd.read_csv('dataset/rumah123_malang.csv')
    df = clean_data(df)
    
    # Open text file for outputs
    with open('hasil_analisis.txt', 'w') as f:
        f.write("=== HASIL ANALISIS REGRESI ===\\n\\n")
        
        # 1. Korelasi
        corr = df[['price_million', 'log_price_million', 'land_size_m2', 'building_size_m2']].corr()
        f.write("1. Matriks Korelasi:\\n")
        f.write(corr.to_string() + "\\n\\n")
        
        # Visualisasi Korelasi
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Heatmap Korelasi')
        plt.tight_layout()
        plt.savefig('output/heatmap_korelasi.png')
        plt.close()
        
        # 2. Regresi Sederhana
        X_sim = df[['building_size_m2']]
        y = df['log_price_million']
        model_sim, mae_sim, rmse_sim, r2_sim = train_evaluate_regression(X_sim, y)
        
        f.write("2. Regresi Linier Sederhana (Predictor: building_size_m2)\\n")
        f.write(f"MAE: {mae_sim:.4f}\\nRMSE: {rmse_sim:.4f}\\nR2 Score: {r2_sim:.4f}\\n\\n")
        
        # 3. Regresi Berganda
        X_mul = df[['land_size_m2', 'building_size_m2']]
        model_mul, mae_mul, rmse_mul, r2_mul = train_evaluate_regression(X_mul, y)
        
        f.write("3. Regresi Linier Berganda (Predictors: land_size_m2, building_size_m2)\\n")
        f.write(f"MAE: {mae_mul:.4f}\\nRMSE: {rmse_mul:.4f}\\nR2 Score: {r2_mul:.4f}\\n\\n")
        
        # 4. Uji Asumsi Klasik
        vif, bp, shapiro_test, resid = check_assumptions(X_mul, y)
        
        f.write("4. Uji Asumsi Klasik (Regresi Berganda)\\n")
        f.write("a. Multikolinearitas (VIF):\\n")
        f.write(vif.to_string() + "\\n\\n")
        f.write(f"b. Heteroskedastisitas (Breusch-Pagan): p-value = {bp[1]:.4f}\\n")
        f.write(f"c. Normalitas Residual (Shapiro-Wilk): p-value = {shapiro_test[1]:.4f}\\n")
        
        # Visualisasi Residual
        plt.figure(figsize=(8, 6))
        sns.histplot(resid, kde=True)
        plt.title('Distribusi Residual')
        plt.tight_layout()
        plt.savefig('output/distribusi_residual.png')
        plt.close()

if __name__ == '__main__':
    main()