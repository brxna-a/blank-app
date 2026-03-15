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

    def padronizar_aba(df_temp, ano):
        df_temp = df_temp.copy()
        df_temp.columns = df_temp.columns.str.strip()

        # cria colunas padrão vazias, se não existirem
        colunas_padrao = [
            "RA", "Fase", "Turma", "Nome", "IAA", "IEG", "IPS", "IDA", "IPV", "IAN",
            "INDE", "Pedra", "Matem", "Portug", "Inglês", "Atingiu PV"
        ]
        for col in colunas_padrao:
            if col not in df_temp.columns:
                df_temp[col] = np.nan

        if ano == "2022":
            # já vem quase no padrão
            if "INDE 22" in df_temp.columns:
                df_temp["INDE"] = df_temp["INDE 22"]
            if "Pedra 22" in df_temp.columns:
                df_temp["Pedra"] = df_temp["Pedra 22"]

        elif ano == "2023":
            # padronizações da aba 2023
            if "INDE 2023" in df_temp.columns:
                df_temp["INDE"] = df_temp["INDE 2023"]
            elif "INDE 23" in df_temp.columns:
                df_temp["INDE"] = df_temp["INDE 23"]

            if "Pedra 2023" in df_temp.columns:
                df_temp["Pedra"] = df_temp["Pedra 2023"]
            elif "Pedra 23" in df_temp.columns:
                df_temp["Pedra"] = df_temp["Pedra 23"]

            if "Mat" in df_temp.columns:
                df_temp["Matem"] = df_temp["Mat"]
            if "Por" in df_temp.columns:
                df_temp["Portug"] = df_temp["Por"]
            if "Ing" in df_temp.columns:
                df_temp["Inglês"] = df_temp["Ing"]

        elif ano == "2024":
            # padronizações da aba 2024
            if "INDE 2024" in df_temp.columns:
                df_temp["INDE"] = df_temp["INDE 2024"]

            if "Pedra 2024" in df_temp.columns:
                df_temp["Pedra"] = df_temp["Pedra 2024"]

            if "Mat" in df_temp.columns:
                df_temp["Matem"] = df_temp["Mat"]
            if "Por" in df_temp.columns:
                df_temp["Portug"] = df_temp["Por"]
            if "Ing" in df_temp.columns:
                df_temp["Inglês"] = df_temp["Ing"]

        df_temp["Ano"] = ano
        return df_temp

    dfs = []
    for sheet in ["PEDE2022", "PEDE2023", "PEDE2024"]:
        ano = sheet.replace("PEDE", "")
        df_temp = pd.read_excel(xls, sheet_name=sheet)
        df_temp = padronizar_aba(df_temp, ano)
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    return df


df = load_data()



# =========================================================
# TRATAMENTO DOS DADOS
# =========================================================

df.columns = df.columns.str.strip()

df = df.drop_duplicates(subset=["RA","Ano"])

df = df.replace(["","NA","-"],np.nan)


cols_numericas = [
    "IAA","IEG","IPS","IDA","IPV","IAN","Matem","Portug","Inglês","INDE"
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
# VISÃO GERAL
# =========================================================

st.header("Visão Geral dos Indicadores")


col1,col2,col3,col4,col5 = st.columns(5)

col1.metric("Total de alunos",df["RA"].nunique())
col2.metric("Registros analisados",len(df))
col3.metric("INDE médio", round(df["INDE"].mean(), 2))
col4.metric("Engajamento médio",round(df["IEG"].mean(),2))
col5.metric("Aprendizagem média",round(df["IDA"].mean(),2))

# =========================================================
# GRÁFICOS DE EVOLUÇÃO (SOMENTE TODOS)
# =========================================================

if ano_selecionado == "Todos":

    st.header("Evolução dos Indicadores ao Longo dos Anos")

    st.caption("""
    Este gráfico apresenta a evolução média dos principais indicadores educacionais ao longo dos anos analisados no programa.

    Os indicadores incluem engajamento dos alunos (IEG), fatores psicossociais (IPS), desempenho acadêmico (IDA), ponto de virada educacional (IPV) e adequação ao nível esperado (IAN).

    A análise da evolução ao longo do tempo permite identificar tendências de melhoria ou queda nesses indicadores, ajudando a avaliar o impacto do programa educacional no desenvolvimento dos alunos.
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

    st.header("Quantidade de alunos acompanhados por ano")

    st.caption("""
    Este gráfico apresenta o número de alunos únicos acompanhados em cada ano do programa Passos Mágicos.
    Ele permite visualizar o crescimento ou variação do número de estudantes atendidos ao longo do tempo.
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
As **Pedras** representam o nível de desenvolvimento educacional do aluno dentro do programa.

Quartzo → maior defasagem educacional  
Ágata → nível intermediário de desenvolvimento  
Ametista → bom desempenho educacional  
Topázio → alunos destaque
""")

fig = px.histogram(df,x="Pedra",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# IAN
# =========================================================

st.header("Adequação ao nível (IAN)")

st.caption("""
O **IAN (Indicador de Adequação ao Nível)** mede se o aluno está no nível educacional esperado para sua idade ou série escolar.

Valores mais baixos indicam maior defasagem educacional, enquanto valores mais altos indicam que o estudante está mais próximo do nível adequado.

A distribuição desse indicador permite identificar o grau de adequação educacional dos alunos e avaliar a presença de possíveis lacunas de aprendizagem.
""")

fig = px.histogram(df,x="IAN",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# DEFASAGEM
# =========================================================

st.header("Classificação de Defasagem Educacional")

st.caption("""
Esta visualização classifica os alunos de acordo com o nível de defasagem educacional com base no indicador de adequação ao nível (IAN).

Os estudantes são agrupados em três categorias principais:

• Alta defasagem educacional  
• Defasagem moderada  
• Nível educacional adequado

Essa classificação facilita a identificação de grupos de alunos que podem demandar maior atenção pedagógica ou estratégias de apoio educacional.
""")

fig = px.histogram(df,x="Nivel Defasagem",color="Nivel Defasagem")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# ENGAJAMENTO VS APRENDIZAGEM
# =========================================================

st.header("Engajamento vs Aprendizagem")

st.caption("""
Este gráfico compara o nível de engajamento dos alunos nas atividades do programa com seu desempenho acadêmico.

O **IEG (Indicador de Engajamento)** representa o nível de participação dos alunos nas atividades educacionais, enquanto o **IDA (Indicador de Desempenho Acadêmico)** mede o resultado educacional obtido.

A análise conjunta desses indicadores permite investigar se alunos mais engajados tendem a apresentar melhor desempenho acadêmico.
""")

fig = px.scatter(df,x="IEG",y="IDA",color="Pedra",opacity=0.6)

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# AUTOAVALIAÇÃO
# =========================================================

st.header("Autoavaliação vs Desempenho")

st.caption("""
O **IAA (Indicador de Autoavaliação)** representa como os próprios alunos percebem seu desempenho educacional.

Ao comparar esse indicador com o **IDA (Indicador de Desempenho Acadêmico)**, é possível avaliar se a percepção dos estudantes sobre seu aprendizado está alinhada com seu desempenho real.

A linha de referência no gráfico representa o ponto em que percepção e desempenho são equivalentes.
""")

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

st.caption("""
O **IPS (Indicador Psicossocial)** mede fatores emocionais, sociais e comportamentais que podem influenciar o processo de aprendizagem dos alunos.

Esses fatores incluem aspectos como motivação, bem-estar emocional e ambiente social.

A análise desse indicador ajuda a compreender como fatores não diretamente acadêmicos podem impactar o desempenho educacional.
""")

fig = px.box(df,x="Pedra",y="IPS",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# PONTO DE VIRADA
# =========================================================

st.header("Ponto de Virada")

st.caption("""
O **IPV (Indicador de Ponto de Virada)** identifica momentos em que o aluno apresenta mudanças relevantes em seu comportamento ou desempenho educacional.

Esses pontos podem indicar fases de evolução significativa, melhoria no engajamento ou mudanças positivas no processo de aprendizagem.

A comparação entre alunos que atingiram ou não esse ponto permite analisar possíveis impactos no desempenho acadêmico.
""")

fig = px.box(df,x="Atingiu PV",y="IDA",color="Atingiu PV")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# NOTAS
# =========================================================

st.header("Notas Escolares")

st.caption("""
Esta análise apresenta a distribuição das notas dos alunos nas principais disciplinas escolares.

A comparação entre as disciplinas permite identificar áreas em que os estudantes apresentam maior desempenho ou maior dificuldade.

Essa informação pode auxiliar na identificação de possíveis lacunas de aprendizagem em áreas específicas do conhecimento.
""")

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

st.caption("""
O **Score Educacional** foi criado para sintetizar o desempenho global dos alunos a partir da combinação de diferentes indicadores educacionais.

Esse índice agrega informações sobre engajamento, aspectos psicossociais, desempenho acadêmico, ponto de virada educacional e adequação ao nível.

A análise desse score permite observar de forma mais integrada o desenvolvimento educacional dos estudantes.
""")

fig = px.histogram(df,x="Score Educacional",color="Pedra")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# CORRELAÇÃO
# =========================================================

st.header("Correlação entre Indicadores")

st.caption("""
A matriz de correlação apresenta o grau de relação entre os principais indicadores educacionais analisados.

Valores próximos de **1** indicam forte relação positiva entre os indicadores, enquanto valores próximos de **0** indicam baixa relação.

Essa análise ajuda a compreender como diferentes fatores educacionais se influenciam mutuamente.
""")

corr = df[["IEG","IPS","IDA","IPV","IAN"]].corr()

fig = px.imshow(corr,text_auto=True,color_continuous_scale="RdBu")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# MACHINE LEARNING
# =========================================================

st.header("Previsão de Risco Educacional")

st.caption(""" Foi desenvolvido um modelo de **Machine Learning (Random Forest)** para prever o risco educacional dos alunos. 

O modelo utiliza indicadores como **engajamento (IEG), desempenho acadêmico (IDA), fatores psicossociais (IPS) e ponto de virada (IPV)**. 

Com base nesses indicadores, o modelo identifica padrões associados ao risco educacional e permite antecipar alunos que podem apresentar dificuldades de aprendizagem, auxiliando na priorização de intervenções pedagógicas. """) 
st.caption(""" O modelo apresentou alta acurácia na identificação do risco educacional. 
Entretanto, é importante considerar que os resultados dependem da definição da variável de risco e da distribuição dos dados no conjunto analisado. """)


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

st.caption(""" O gráfico a seguir mostra a **importância dos indicadores utilizados pelo modelo de Machine Learning para prever o risco educacional**. 

Cada barra representa o quanto um indicador contribui para a capacidade do modelo de identificar alunos em risco.
Indicadores com maior importância têm maior impacto na previsão e podem ser considerados fatores críticos para monitoramento e intervenção educacional. """)


fig.update_traces(texttemplate='%{text:.2f}',textposition="outside")

st.plotly_chart(fig,use_container_width=True)

# =========================================================
# INSIGHTS
# =========================================================

st.header("Insights")

st.caption("""
Esta seção apresenta algumas relações observadas entre os indicadores educacionais analisados.

A análise das correlações entre engajamento, fatores psicossociais e desempenho acadêmico ajuda a compreender melhor quais fatores podem estar mais associados ao sucesso educacional dos alunos.
""")

corr1 = df[["IEG","IDA"]].corr().iloc[0,1]
corr2 = df[["IPS","IDA"]].corr().iloc[0,1]

st.write("Correlação Engajamento x Desempenho:",round(corr1,2))
st.write("Correlação Psicossocial x Desempenho:",round(corr2,2))