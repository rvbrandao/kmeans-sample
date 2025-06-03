import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Simulação de dados de comportamento de usuários de apostas
np.random.seed(42)
n_users = 1000  # número de usuários

# Simulando os vetores
data = pd.DataFrame({
    'user_id': [f'U{i:03d}' for i in range(n_users)],
    'total_bets': np.random.gamma(2, 1500, n_users),  # apostas totais
    'freq_bets': np.random.poisson(25, n_users),      # frequência de apostas no mês
    'perc_sports': np.clip(np.random.normal(0.6, 0.3, n_users), 0, 1),  # % em esportes
    'perc_casino': 0,  # será 1 - perc_sports
    'avg_deposit': np.random.gamma(2, 400, n_users),
    'avg_withdraw': np.random.gamma(1.5, 300, n_users),
    'active_days': np.random.randint(5, 31, n_users)
})

data['perc_casino'] = 1 - data['perc_sports']  # complementar

# Normalização
features = ['total_bets', 'freq_bets', 'perc_sports', 'perc_casino',
            'avg_deposit', 'avg_withdraw', 'active_days']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Aplicando KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['cluster'] = clusters

# Visualização 2D usando PCA para simplificar
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
data['pca1'] = X_pca[:, 0]
data['pca2'] = X_pca[:, 1]

# Plot dos clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='pca1', y='pca2', hue='cluster', palette='Set2', s=60)
plt.title('Clusters de usuários por comportamento de apostas (KMeans)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

data.to_csv('user_behavior_vector.csv', index=False)
