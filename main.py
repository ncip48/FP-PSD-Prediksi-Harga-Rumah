import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_data, train_evaluate_regression, check_assumptions


def main():
    print("========================================")
    print("Starting Regression Analysis")
    print("========================================")

    # Load data
    print("\n[1/7] Loading dataset...")
    df = pd.read_csv('dataset/rumah123_malang.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    print("\n[2/7] Cleaning dataset...")
    df = clean_data(df)
    print(f"Dataset cleaned. Shape after cleaning: {df.shape}")

    # Open text file for outputs
    print("\nOpening output file...")
    with open('hasil_analisis.txt', 'w') as f:
        f.write("=== HASIL ANALISIS REGRESI ===\n\n")

        # 1. Korelasi
        print("\n[3/7] Calculating correlation matrix...")
        corr = df[
            [
                'price_million',
                'building_size_m2',
                'land_size_m2',
                'log_price_million'
            ]
        ].corr()

        print(corr)

        f.write("1. Matriks Korelasi:\n")
        f.write(corr.to_string() + "\n\n")

        print("Saving correlation heatmap...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Heatmap Korelasi')
        plt.tight_layout()
        plt.savefig('output/heatmap_korelasi.png')
        plt.close()
        print("✓ Heatmap saved to output/heatmap_korelasi.png")

        # 2. Regresi Sederhana
        print("\n[4/7] Training Simple Linear Regression...")
        X_sim = df[['building_size_m2']]
        y = df['log_price_million']

        model_sim, mae_sim, rmse_sim, r2_sim = train_evaluate_regression(X_sim, y)

        print(f"MAE  : {mae_sim:.4f}")
        print(f"RMSE : {rmse_sim:.4f}")
        print(f"R²   : {r2_sim:.4f}")

        f.write("2. Regresi Linier Sederhana (Predictor: building_size_m2)\n")
        f.write(
            f"MAE: {mae_sim:.4f}\n"
            f"RMSE: {rmse_sim:.4f}\n"
            f"R2 Score: {r2_sim:.4f}\n\n"
        )

        # 3. Regresi Berganda
        print("\n[5/7] Training Multiple Linear Regression...")
        X_mul = df[['land_size_m2', 'building_size_m2']]

        model_mul, mae_mul, rmse_mul, r2_mul = train_evaluate_regression(X_mul, y)

        print(f"MAE  : {mae_mul:.4f}")
        print(f"RMSE : {rmse_mul:.4f}")
        print(f"R²   : {r2_mul:.4f}")

        f.write("3. Regresi Linier Berganda (Predictors: land_size_m2, building_size_m2)\n")
        f.write(
            f"MAE: {mae_mul:.4f}\n"
            f"RMSE: {rmse_mul:.4f}\n"
            f"R2 Score: {r2_mul:.4f}\n\n"
        )

        # 4. Uji Asumsi Klasik
        print("\n[6/7] Running classical assumption tests...")
        vif, bp, shapiro_test, resid = check_assumptions(X_mul, y)

        print("\nVariance Inflation Factor (VIF):")
        print(vif)

        print(f"\nBreusch-Pagan p-value : {bp[1]:.4f}")
        print(f"Shapiro-Wilk p-value  : {shapiro_test[1]:.4f}")

        f.write("4. Uji Asumsi Klasik (Regresi Berganda)\n")
        f.write("a. Multikolinearitas (VIF):\n")
        f.write(vif.to_string() + "\n\n")
        f.write(f"b. Heteroskedastisitas (Breusch-Pagan): p-value = {bp[1]:.4f}\n")
        f.write(f"c. Normalitas Residual (Shapiro-Wilk): p-value = {shapiro_test[1]:.4f}\n")

        # Visualisasi Residual
        print("\n[7/7] Saving residual distribution plot...")
        plt.figure(figsize=(8, 6))
        sns.histplot(resid, kde=True)
        plt.title('Distribusi Residual')
        plt.tight_layout()
        plt.savefig('output/distribusi_residual.png')
        plt.close()

        print("✓ Residual plot saved to output/distribusi_residual.png")

    print("\n========================================")
    print("Analysis completed successfully!")
    print("Results saved to:")
    print(" - hasil_analisis.txt")
    print(" - output/heatmap_korelasi.png")
    print(" - output/distribusi_residual.png")
    print("========================================")


if __name__ == '__main__':
    main()