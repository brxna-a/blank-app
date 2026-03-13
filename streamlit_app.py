import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(layout="wide")

st.title("📊 Dashboard Educacional - Passos Mágicos")

st.write("""
Este dashboard apresenta uma análise dos indicadores educacionais dos alunos do programa **Passos Mágicos**.
Os gráficos exploram engajamento, desempenho acadêmico, fatores psicossociais e risco educacional.
""")

# =============================
# CARREGAR DADOS
# =============================

@st.cache_data
def load_data():
    df = pd.read_excel("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
    return df

df = load_data()

# =============================
# LIMPEZA
# =============================

df.columns = df.columns.str.strip()

df["Pedra"] = df["Pedra 22"].fillna(df["Pedra 21"]).fillna(df["Pedra 20"])

cols_numericas = [
"IAA","IEG","IPS","IDA","IPV","IAN","Matem","Portug","Inglês","INDE 22"
]

for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================
# VISÃO GERAL
# =============================

st.header("Visão Geral dos Indicadores")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total de alunos",len(df))
col2.metric("INDE médio",round(df["INDE 22"].mean(),2))
col3.metric("Engajamento médio",round(df["IEG"].mean(),2))
col4.metric("Aprendizagem média",round(df["IDA"].mean(),2))

# =============================
# DISTRIBUIÇÃO DAS PEDRAS
# =============================

st.header("Distribuição das Pedras")

st.caption("""
As **Pedras** representam o nível de desenvolvimento educacional do aluno.

Quartzo → maior defasagem  
Ágata → nível intermediário  
Ametista → bom desempenho  
Topázio → alunos destaque
""")

fig = px.histogram(
    df,
    x="Pedra",
    color="Pedra",
    title="Distribuição dos alunos por nível educacional"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# ADEQUAÇÃO AO NÍVEL
# =============================

st.header("Adequação ao nível (IAN)")

st.caption("""
O **IAN** mede se o aluno está no nível educacional adequado para sua série/idade.
Valores menores indicam maior defasagem educacional.
""")

fig = px.histogram(
    df,
    x="IAN",
    color="Pedra",
    title="Distribuição da adequação educacional"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# ENGAJAMENTO VS APRENDIZAGEM
# =============================

st.header("Engajamento vs Aprendizagem")

st.caption("""
O **IEG (Indicador de Engajamento)** mede participação do aluno nas atividades.

O **IDA (Indicador de Aprendizagem)** mede o desempenho acadêmico.

O gráfico mostra se alunos mais engajados tendem a apresentar melhor desempenho.
""")

fig = px.scatter(
    df,
    x="IEG",
    y="IDA",
    color="Pedra",
    opacity=0.6,
    trendline="ols"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# AUTOAVALIAÇÃO VS DESEMPENHO
# =============================

st.header("Autoavaliação vs Desempenho")

st.caption("""
O **IAA (Indicador de Autoavaliação)** representa como o aluno percebe seu próprio desempenho.

Comparando com o **IDA**, podemos avaliar se os alunos têm percepção realista da própria aprendizagem.
""")

fig = px.scatter(
    df,
    x="IAA",
    y="IDA",
    color="Pedra",
    opacity=0.6,
    trendline="ols"
)

fig.add_shape(
    type="line",
    x0=0,y0=0,
    x1=10,y1=10,
    line=dict(color="white",dash="dash")
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# INDICADOR PSICOSSOCIAL
# =============================

st.header("Indicador Psicossocial")

st.caption("""
O **IPS** mede fatores emocionais e sociais que influenciam o aprendizado,
como motivação, ambiente familiar e bem-estar do aluno.
""")

fig = px.box(
    df,
    x="Pedra",
    y="IPS",
    color="Pedra",
    title="Distribuição do indicador psicossocial por nível educacional"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# PONTO DE VIRADA
# =============================

st.header("Ponto de Virada")

st.caption("""
O **IPV (Indicador de Ponto de Virada)** identifica mudanças significativas no comportamento ou desempenho do aluno.

Essas mudanças podem indicar melhora ou risco de queda no desempenho.
""")

fig = px.box(
    df,
    x="Atingiu PV",
    y="IDA",
    color="Atingiu PV",
    title="Relação entre ponto de virada e desempenho acadêmico"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# NOTAS ESCOLARES
# =============================

st.header("Notas Escolares")

st.caption("""
Comparação das notas nas disciplinas principais para identificar em quais áreas os alunos apresentam maior dificuldade.
""")

df_notas = df.melt(
    value_vars=["Matem","Portug","Inglês"],
    var_name="Disciplina",
    value_name="Nota"
)

fig = px.box(
    df_notas,
    x="Disciplina",
    y="Nota",
    color="Disciplina",
    title="Distribuição das notas por disciplina"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# CORRELAÇÃO ENTRE INDICADORES
# =============================

st.header("Correlação entre Indicadores")

st.caption("""
A matriz de correlação mostra como os indicadores educacionais se relacionam entre si.
Valores próximos de 1 indicam relação forte positiva.
""")

corr = df[["IEG","IPS","IDA","IPV","IAN"]].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Correlação entre indicadores educacionais"
)

st.plotly_chart(fig,use_container_width=True)

# =============================
# MACHINE LEARNING
# =============================

st.header("Previsão de Risco Educacional")

st.caption("""
Utilizamos um modelo de Machine Learning para identificar alunos com maior probabilidade
de apresentar dificuldades educacionais.
""")

df_ml = df.dropna(subset=["IAN","IEG","IPS","IDA","IPV"])

df_ml["risco"] = (
(df_ml["IAN"] < 5) |
(df_ml["IDA"] < 5)
).astype(int)

features = ["IEG","IPS","IDA","IPV"]

X = df_ml[features]
y = df_ml["risco"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.3,random_state=42
)

model = RandomForestClassifier(
class_weight="balanced",
random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

st.metric("Acurácia do modelo",round(acc,2))

st.text(classification_report(y_test,pred))

# importância das variáveis

importance = pd.DataFrame({
"Variável":features,
"Importância":model.feature_importances_
}).sort_values("Importância",ascending=False)

fig = px.bar(
importance,
x="Variável",
y="Importância",
color="Variável",
text="Importância",
title="Indicadores que mais influenciam o risco educacional"
)

fig.update_traces(texttemplate='%{text:.2f}',textposition="outside")

st.plotly_chart(fig,use_container_width=True)

# =============================
# INSIGHTS
# =============================

st.header("Insights")

corr1 = df[["IEG","IDA"]].corr().iloc[0,1]
corr2 = df[["IPS","IDA"]].corr().iloc[0,1]

st.write("Correlação Engajamento x Desempenho:",round(corr1,2))
st.write("Correlação Psicossocial x Desempenho:",round(corr2,2))