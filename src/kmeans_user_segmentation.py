# kmeans_user_segmentation.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import datetime

# === Passo 1: Carregar dados ===
# Simulando leitura de BigQuery. Na prática, você carregaria via 'pandas_gbq.read_gbq' ou outro meio.
data = pd.read_csv("user_behavior_vector.csv", header=0)  # Supondo que já tenha gerado essa tabela
user_ids = data["user_id"]

# === Passo 2: Selecionar e normalizar as features ===
features = [
    "total_bets", "freq_bets", "perc_sports", "perc_casino",
    "avg_deposit", "avg_withdraw", "active_days"
]
X = data[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Passo 3: Treinar o modelo KMeans ===
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
kmeans.fit(X_scaled)

# === Passo 4: Atribuir clusters e nomear perfis (opcional) ===
data["cluster_behavior"] = kmeans.predict(X_scaled)

# Rótulos interpretáveis (vários métodos possíveis)
cluster_names = {
    0: "Fanático por Futebol",
    1: "High Roller Cassino",
    2: "Casual Mix",
    3: "Recente ou Inativo"
}
data["cluster_name"] = data["cluster_behavior"].map(cluster_names)

# === Passo 5: Salvar modelo e scaler para reutilização futura ===
today = datetime.date.today().isoformat()
joblib.dump(kmeans, f"kmeans_model_{today}.pkl")
joblib.dump(scaler, f"scaler_{today}.pkl")

# === Passo 6: Exportar resultado para CSV (ou BigQuery) ===
data[["user_id", "cluster_behavior", "cluster_name"] + features].to_csv("clustered_users.csv", index=False)
print("Clusterização concluída. Resultados salvos em clustered_users.csv")
