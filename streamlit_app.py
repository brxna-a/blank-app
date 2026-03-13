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

Criamos um score educacional agregando múltiplos indicadores para sintetizar o desempenho global do aluno e utilizamos o dataset PEDE para todo o desenvolvimento deste dashboard.
""")

# =========================================================
# CARREGAR DADOS
# =========================================================

@st.cache_data
def load_data():

    excel = pd.ExcelFile("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")

    df2022 = pd.read_excel(excel, sheet_name="PEDE2022")
    df2023 = pd.read_excel(excel, sheet_name="PEDE2023")
    df2024 = pd.read_excel(excel, sheet_name="PEDE2024")

    df2022["Ano"] = "2022"
    df2023["Ano"] = "2023"
    df2024["Ano"] = "2024"

    df = pd.concat([df2022, df2023, df2024], ignore_index=True)

    return df


df = load_data()

# =========================================================
# FILTRO DE ANO
# =========================================================

st.sidebar.header("Filtros")

ano_selecionado = st.sidebar.selectbox(
    "Selecionar ano",
    ["Todos", "2022", "2023", "2024"]
)

if ano_selecionado != "Todos":
    df = df[df["Ano"] == ano_selecionado]

# =========================================================
# TRATAMENTO DOS DADOS
# =========================================================

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

df = df.replace(["", "NA", "-"], np.nan)

df["Pedra"] = df["Pedra 22"].fillna(df["Pedra 21"]).fillna(df["Pedra 20"])

cols_numericas = [
"IAA","IEG","IPS","IDA","IPV","IAN","Matem","Portug","Inglês","INDE 22"
]

for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

indicadores = ["IAA","IEG","IPS","IDA","IPV","IAN"]

for col in indicadores:
    df[col] = df[col].clip(0,10)

# =========================================================
# CRIAÇÃO DE NOVAS VARIÁVEIS
# =========================================================

df["Nivel Defasagem"] = pd.cut(
    df["IAN"],
    bins=[0,4,7,10],
    labels=["Alta defasagem","Defasagem moderada","Adequado"]
)

df["Score Educacional"] = (
    df["IEG"]*0.2 +
    df["IPS"]*0.2 +
    df["IDA"]*0.3 +
    df["IPV"]*0.2 +
    df["IAN"]*0.1
)

# =========================================================
# VISÃO GERAL
# =========================================================

st.header("Visão Geral dos Indicadores")

col1,col2,col3,col4,col5 = st.columns(5)

# alunos únicos
total_alunos = df["RA"].nunique()

# alunos novos (primeira vez no dataset)
primeiro_ano = df.groupby("RA")["Ano"].min()
alunos_novos = (primeiro_ano == ano_selecionado).sum() if ano_selecionado != "Todos" else len(primeiro_ano)

col1.metric("Alunos únicos", total_alunos)

col2.metric(
    "Registros no dataset",
    len(df)
)

col3.metric(
    "INDE médio",
    round(df["INDE 22"].mean(),2)
)

col4.metric(
    "Engajamento médio",
    round(df["IEG"].mean(),2)
)

col5.metric(
    "Aprendizagem média",
    round(df["IDA"].mean(),2)
)

if ano_selecionado == "Todos":
    # =========================================================
    # EVOLUÇÃO DOS INDICADORES
    # =========================================================

    st.header("Evolução dos Indicadores ao Longo dos Anos")

    st.caption("""
    Este gráfico apresenta a evolução média dos principais indicadores educacionais ao longo dos anos do programa.
    Ele permite observar tendências de melhoria ou queda nos indicadores de engajamento, aprendizagem e fatores psicossociais.
    """)

    evolucao = df.groupby("Ano", as_index=False)[["IEG","IPS","IDA","IPV","IAN"]].mean()

    # garantir texto/categoria
    evolucao["Ano"] = evolucao["Ano"].astype(str)

    # ordenar corretamente
    ordem_anos = ["2022", "2023", "2024"]

    evolucao["Ano"] = pd.Categorical(
        evolucao["Ano"],
        categories=ordem_anos,
        ordered=True
    )

    evolucao = evolucao.sort_values("Ano")

    fig = px.line(
        evolucao,
        x="Ano",
        y=["IEG","IPS","IDA","IPV","IAN"],
        markers=True,
        category_orders={"Ano": ordem_anos}
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=ordem_anos,
        title="Ano"
    )

    st.plotly_chart(fig,use_container_width=True)
    

    # ==================================
    # ALUNOS ACOMPANHADOS
    # ==================================
    st.header("Quantidade de alunos acompanhados por ano")

    st.caption("""
    Este gráfico mostra quantos alunos únicos foram acompanhados em cada ano do programa.
    """)

    alunos_ano = (
        df.groupby("Ano", as_index=False)["RA"]
        .nunique()
        .rename(columns={"RA": "Quantidade de alunos"})
    )

    alunos_ano["Ano"] = alunos_ano["Ano"].astype(str)

    ordem_anos = ["2022", "2023", "2024"]

    alunos_ano["Ano"] = pd.Categorical(
        alunos_ano["Ano"],
        categories=ordem_anos,
        ordered=True
    )

    alunos_ano = alunos_ano.sort_values("Ano")

    fig = px.bar(
        alunos_ano,
        x="Ano",
        y="Quantidade de alunos",
        text="Quantidade de alunos",
        title="• Quantidade de alunos acompanhados por ano",
        category_orders={"Ano": ordem_anos}
    )

    fig.update_traces(textposition="outside")

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=ordem_anos,
        title="Ano"
    )

    fig.update_yaxes(title="Número de alunos")

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# DISTRIBUIÇÃO DAS PEDRAS
# =========================================================

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
    title="• Distribuição dos alunos por nível educacional"
)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# ADEQUAÇÃO AO NÍVEL
# =========================================================

st.header("Adequação ao nível (IAN)")

st.caption("""
O **IAN** mede se o aluno está no nível educacional adequado para sua série/idade.
Valores menores indicam maior defasagem educacional.
""")

fig = px.histogram(
    df,
    x="IAN",
    color="Pedra",
    title="• Distribuição da adequação educacional"
)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# ENGAJAMENTO VS APRENDIZAGEM
# =========================================================

st.header("Engajamento vs Aprendizagem")

st.caption("""
O **IEG (Indicador de Engajamento)** mede participação do aluno nas atividades.

O **IDA (Indicador de Aprendizagem)** mede o desempenho acadêmico.
""")

fig = px.scatter(
    df,
    x="IEG",
    y="IDA",
    color="Pedra",
    opacity=0.6
)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# AUTOAVALIAÇÃO VS DESEMPENHO
# =========================================================

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
    opacity=0.6
)

fig.add_shape(
    type="line",
    x0=0,y0=0,
    x1=10,y1=10,
    line=dict(color="white",dash="dash")
)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# MACHINE LEARNING
# =========================================================

st.header("Previsão de Risco Educacional")

st.caption("""
Foi desenvolvido um modelo de **Machine Learning (Random Forest)** para prever o risco educacional dos alunos.

O modelo utiliza indicadores como **engajamento (IEG), desempenho acadêmico (IDA), fatores psicossociais (IPS) e ponto de virada (IPV)**.

Com base nesses indicadores, o modelo identifica padrões associados ao risco educacional e permite antecipar alunos que podem apresentar dificuldades de aprendizagem, auxiliando na priorização de intervenções pedagógicas.
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

st.metric(
    "Acurácia do modelo",
    f"{acc:.2%}"
)

st.text(classification_report(y_test,pred))

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
title="• Indicadores que mais influenciam o risco educacional"
)

fig.update_traces(texttemplate='%{text:.2f}',textposition="outside")

st.plotly_chart(fig,use_container_width=True)