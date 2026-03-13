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

    xls = pd.ExcelFile("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")

    sheets = ["PEDE2022","PEDE2023","PEDE2024"]

    dfs = []

    for sheet in sheets:

        df_temp = pd.read_excel(xls, sheet_name=sheet)

        ano = sheet.replace("PEDE","")

        df_temp["Ano"] = ano

        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    return df


df = load_data()

# =========================================================
# FILTRO DE ANO
# =========================================================

st.sidebar.header("Filtros")

anos = ["Todos","2022","2023","2024"]

ano_selecionado = st.sidebar.selectbox(
    "Selecione o ano",
    anos
)

if ano_selecionado != "Todos":
    df = df[df["Ano"] == ano_selecionado]

# =========================================================
# TRATAMENTO DOS DADOS
# =========================================================

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

df = df.replace(["","NA","-"],np.nan)

df["Pedra"] = df["Pedra 22"].fillna(df["Pedra 21"]).fillna(df["Pedra 20"])

cols_numericas = [
"IAA","IEG","IPS","IDA","IPV","IAN","Matem","Portug","Inglês","INDE 22"
]

for col in cols_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col],errors="coerce")

indicadores = ["IAA","IEG","IPS","IDA","IPV","IAN"]

for col in indicadores:
    df[col] = df[col].clip(0,10)

# =========================================================
# NOVAS VARIÁVEIS
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

st.caption("""
Resumo geral dos principais indicadores educacionais analisados no dashboard.
Essas métricas apresentam uma visão consolidada do desempenho dos alunos acompanhados pelo programa.
""")

col1,col2,col3,col4,col5 = st.columns(5)

# alunos únicos
col1.metric(
    "Total de alunos",
    df["RA"].nunique()
)

# registros no dataset (histórico)
col2.metric(
    "Registros analisados",
    len(df)
)

# INDE
col3.metric(
    "INDE médio",
    round(df["INDE 22"].mean(),2)
)

# engajamento
col4.metric(
    "Engajamento médio",
    round(df["IEG"].mean(),2)
)

# aprendizagem
col5.metric(
    "Aprendizagem média",
    round(df["IDA"].mean(),2)
)

# =========================================================
# GRÁFICOS DE EVOLUÇÃO (SOMENTE TODOS)
# =========================================================

if ano_selecionado == "Todos":

    st.header("Evolução dos Indicadores ao Longo dos Anos")

    st.caption("""
    Este gráfico apresenta a evolução média dos principais indicadores educacionais ao longo dos anos do programa.
    """)

    evolucao = df.groupby("Ano",as_index=False)[["IEG","IPS","IDA","IPV","IAN"]].mean()

    ordem = ["2022","2023","2024"]

    evolucao["Ano"] = pd.Categorical(
        evolucao["Ano"].astype(str),
        categories=ordem,
        ordered=True
    )

    evolucao = evolucao.sort_values("Ano")

    fig = px.line(
        evolucao,
        x="Ano",
        y=["IEG","IPS","IDA","IPV","IAN"],
        markers=True
    )

    fig.update_xaxes(type="category")

    st.plotly_chart(fig,use_container_width=True)

    # ------------------------------------------------------

    st.header("Quantidade de alunos acompanhados por ano")

    st.caption("""
    Este gráfico mostra quantos alunos únicos foram acompanhados em cada ano do programa.
    """)

    alunos = df.groupby("Ano")["RA"].nunique().reset_index()

    alunos["Ano"] = pd.Categorical(
        alunos["Ano"].astype(str),
        categories=ordem,
        ordered=True
    )

    alunos = alunos.sort_values("Ano")

    fig = px.bar(
        alunos,
        x="Ano",
        y="RA",
        text="RA",
        title="• Quantidade de alunos acompanhados por ano"
    )

    fig.update_traces(textposition="outside")

    fig.update_xaxes(type="category")

    st.plotly_chart(fig,use_container_width=True)

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

fig = px.histogram(df,x="Pedra",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# IAN
# =========================================================

st.header("Adequação ao nível (IAN)")

st.caption("""
O **IAN** mede se o aluno está no nível educacional adequado para sua série/idade.
""")

fig = px.histogram(df,x="IAN",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# DEFASAGEM
# =========================================================

st.header("Classificação de Defasagem Educacional")

fig = px.histogram(df,x="Nivel Defasagem",color="Nivel Defasagem")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# ENGAJAMENTO VS APRENDIZAGEM
# =========================================================

st.header("Engajamento vs Aprendizagem")

st.caption("""
O **IEG** mede participação nas atividades e o **IDA** mede o desempenho acadêmico.
""")

fig = px.scatter(df,x="IEG",y="IDA",color="Pedra",opacity=0.6)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# AUTOAVALIAÇÃO
# =========================================================

st.header("Autoavaliação vs Desempenho")

fig = px.scatter(df,x="IAA",y="IDA",color="Pedra",opacity=0.6)

fig.add_shape(
    type="line",
    x0=0,y0=0,
    x1=10,y1=10,
    line=dict(color="white",dash="dash")
)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# PSICOSSOCIAL
# =========================================================

st.header("Indicador Psicossocial")

fig = px.box(df,x="Pedra",y="IPS",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# PONTO DE VIRADA
# =========================================================

st.header("Ponto de Virada")

fig = px.box(df,x="Atingiu PV",y="IDA",color="Atingiu PV")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# NOTAS
# =========================================================

st.header("Notas Escolares")

df_notas = df.melt(
    value_vars=["Matem","Portug","Inglês"],
    var_name="Disciplina",
    value_name="Nota"
)

fig = px.box(df_notas,x="Disciplina",y="Nota",color="Disciplina")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# SCORE EDUCACIONAL
# =========================================================

st.header("Score Educacional Geral")

fig = px.histogram(df,x="Score Educacional",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# CORRELAÇÃO
# =========================================================

st.header("Correlação entre Indicadores")

corr = df[["IEG","IPS","IDA","IPV","IAN"]].corr()

fig = px.imshow(corr,text_auto=True,color_continuous_scale="RdBu")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# MACHINE LEARNING
# =========================================================

st.header("Previsão de Risco Educacional")

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

st.metric("Acurácia do modelo",f"{acc:.2%}")

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
    text="Importância"
)

fig.update_traces(texttemplate='%{text:.2f}',textposition="outside")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# INSIGHTS
# =========================================================

st.header("Insights")

corr1 = df[["IEG","IDA"]].corr().iloc[0,1]
corr2 = df[["IPS","IDA"]].corr().iloc[0,1]

st.write("Correlação Engajamento x Desempenho:",round(corr1,2))
st.write("Correlação Psicossocial x Desempenho:",round(corr2,2))