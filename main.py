import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語対応
from sklearn.preprocessing import MinMaxScaler

# データの読み込み
import os

# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# ファイルへの相対パスを作成
file_path = os.path.join(current_dir, 'pokemon_eng2.txt')

# pandasでCSVファイルを読み込む
pokemon_df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')


type_mapping = {
    'くさ': 1,
    'でんき': 2,
    'ほのお': 3,
    'あく': 4,
    'いわ': 5,
    'エスパー': 6,
    'かくとう': 7,
    'ゴースト': 8,
    'こおり': 9,
    'じめん': 10,
    'どく': 11,
    'ドラゴン': 12,
    'ノーマル': 13,
    'はがね': 14,
    'ひこう': 15,
    'フェアリー': 16,
    'みず': 17,
    'むし': 18
}

# 数値をタイプ名称に変換するための辞書
reverse_type_mapping = {v: k for k, v in type_mapping.items()}

selected_columns = ['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']

# データのスケーリング
# データのスケーリング
scaler = MinMaxScaler()
scaler.fit(pokemon_df[selected_columns])
scaler.feature_names_in_ = None
pokemon_scaled = scaler.transform(pokemon_df[selected_columns])

# KNNモデルの訓練
knn_model = NearestNeighbors(n_neighbors=5)
knn_model.fit(pokemon_scaled)

# Streamlitアプリのメイン部分
st.title('にてるポケモンⅤ')

# ユーザー入力
st.sidebar.header('希望のポケモンステータスを入力してください')
hp = st.sidebar.slider('H.P. - ♥♥♥', 0, int(pokemon_df['HP'].max()))
atk = st.sidebar.slider('攻撃 - Attack', 0, int(pokemon_df['Attack'].max()))
def_ = st.sidebar.slider('防御 - Defense', 0, int(pokemon_df['Defense'].max()))
sp_atk = st.sidebar.slider('特殊攻撃 - Special Attack', 0, int(pokemon_df['Special Attack'].max()))
sp_def = st.sidebar.slider('特殊防御 - Special Defense', 0, int(pokemon_df['Special Defense'].max()))
spd = st.sidebar.slider('素早さ - Speed', 0, int(pokemon_df['Speed'].max()))


user_input = [hp, atk, def_, sp_atk, sp_def, spd]

# ユーザー入力のスケーリング
user_input_scaled = scaler.transform([user_input])

# KNNモデルを使用して類似のポケモンを検索
_, indices = knn_model.kneighbors(user_input_scaled)
similar_pokemon = pokemon_df.iloc[indices[0]]

# 結果の表示

for _, row in similar_pokemon.iterrows():
    # ポケモン名とタイプを一緒に表示
    st.subheader(f"{row['name']} - タイプ: {reverse_type_mapping[row['type']]}")
    
    col1, col2 = st.columns(2)  # 2つのカラムを作成
    
    # 左のカラムに画像を追加
    col1.image(row['URL'], use_column_width=True)
    
    # 右のカラムにレーダーチャートを追加
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    labels = np.array(selected_columns)
    stats = row[labels].values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]
    labels = np.append(labels, labels[0])  # ラベルも1つ追加する
    ax.plot(angles, stats, color='blue', linewidth=2)
    ax.fill(angles, stats, color='blue', alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    col2.pyplot(fig)
