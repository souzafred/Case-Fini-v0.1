import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Inteligência Comercial",
    page_icon="📊",
    layout="wide"
)

# CSS customizado
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Carregamento de CSVs com cache
@st.cache_data
def load_data(uploaded_file):
    try:
        content = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(content, sep=";", decimal=",", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Formatação de mês
def format_month(s):
    m = {'01':'Jan','02':'Fev','03':'Mar','04':'Abr','05':'Mai','06':'Jun',
         '07':'Jul','08':'Ago','09':'Set','10':'Out','11':'Nov','12':'Dez'}
    if isinstance(s,str) and len(s)>=7:
        return f"{m[s[5:7]]}/{s[2:4]}"
    return s

# Título
st.markdown("""
<div style="background: linear-gradient(135deg, #ec4899, #8b5cf6);
            padding:2rem; border-radius:0.5rem; margin-bottom:2rem;">
  <h1 style="color:white;">Dashboard de Inteligência Comercial</h1>
  <p style="color:white;opacity:0.9;">FMCG Doces e Guloseimas</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload de Dados")
    in_file  = st.file_uploader("Sell In",  type="csv")
    out_file = st.file_uploader("Sell Out", type="csv")
    if in_file and out_file:
        st.success("Arquivos carregados!")

sell_in  = load_data(in_file)   if in_file  else None
sell_out = load_data(out_file)  if out_file else None

tab1, tab2, tab3 = st.tabs([
    "📊 Análise de Vendas & KPIs",
    "🔮 Previsão de Vendas",
    "💡 Recomendações Estratégicas"
])

with tab1:
    if sell_in is None or sell_out is None:
        st.warning("⚠️ Faça upload de ambos os arquivos para ver o dashboard.")
    else:
        # --- Ajuste de datas e mês
        sell_in['Data_SellIn'] = pd.to_datetime(sell_in['Data_SellIn'], errors='coerce')
        sell_in['Month']       = sell_in['Data_SellIn'].dt.to_period('M').astype(str)
        sell_out['Data']       = pd.to_datetime(sell_out['Data'], errors='coerce')
        sell_out['Month']      = sell_out['Data'].dt.to_period('M').astype(str)

        # --- KPIs agregados
        total_in   = sell_in['Valor_Total'].sum()
        volume_in  = sell_in['Quantidade'].sum()
        ticket_in  = total_in/volume_in if volume_in else 0

        total_out  = sell_out['Valor_SellThrough'].sum()
        volume_out = sell_out['Unidades_SellThrough'].sum()
        ticket_out = total_out/volume_out if volume_out else 0

        rate = (total_out/total_in*100) if total_in else 0

        def pct_growth(df, col):
            m = df.groupby('Month')[col].sum().sort_index()
            if len(m)>=6:
                return ((m[-3:].sum()-m[-6:-3].sum())/m[-6:-3].sum()*100)
            return 0
        growth_in  = pct_growth(sell_in,   'Valor_Total')
        growth_out = pct_growth(sell_out, 'Valor_SellThrough')

        regions = sell_in['Regiao'].nunique()

        # --- Métricas (TOTAL EM MILHÕES)
        st.markdown("### Sell In")
        c1, c2, c3, c4 = st.columns(4)

        # Total em milhões, 1 casa decimal
        c1.metric(
            "Total (Mi R$)",
            f"{total_in/1e6:.1f} Mi"
        )
        c2.metric(
            "Volume (Cx)",
            f"{volume_in:,.0f}".replace(",", ".")
        )
        c3.metric(
            "Ticket Médio",
            f"R$ {ticket_in:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        cres_in_str = f"{growth_in:,.1f}".replace(".", ",") + "%"
        c4.metric(
            "Crescimento",
            cres_in_str,
            delta=cres_in_str,
            delta_color="normal" if growth_in > 0 else "inverse"
        )

        st.markdown("### Sell Out")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total (Mi R$)",
            f"{total_out/1e6:.1f} Mi"
        )
        c2.metric(
            "Volume (Unid)",
            f"{volume_out:,.0f}".replace(",", ".")
        )
        c3.metric(
            "Ticket Médio",
            f"R$ {ticket_out:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        cres_out_str = f"{growth_out:,.1f}".replace(".", ",") + "%"
        c4.metric(
            "Crescimento",
            cres_out_str,
            delta=cres_out_str,
            delta_color="normal" if growth_out > 0 else "inverse"
        )

        # --- KPI Geral Sell-Through
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    color:white;padding:1rem;border-radius:0.5rem;margin:2rem 0;">
          <div style="display:flex;justify-content:space-between">
            <div>
              <h3>🎯 Sell-Through Rate</h3>
              <p style="font-size:2rem;margin:0;">{rate:.1f}%</p>
              <small>Meta: 85%</small>
            </div>
            <div style="text-align:right">
              <small>Regiões</small>
              <p style="font-size:1.5rem;margin:0;">{regions}</p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Gráficos (mantém como antes)
        mi = sell_in.groupby('Month')['Valor_Total'].sum().reset_index()
        mo = sell_out.groupby('Month')['Valor_SellThrough'].sum().reset_index()
        dfm = mi.merge(mo, on='Month', how='outer').fillna(0)
        dfm['Rate'] = (dfm['Valor_SellThrough']/dfm['Valor_Total']*100).clip(0,150)
        dfm = dfm[dfm['Month']>='2023-01']
        dfm['M'] = dfm['Month'].apply(format_month)

        a1, a2 = st.columns(2)
        with a1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dfm['M'], y=dfm['Rate'],
                mode='lines+markers+text',
                text=[f"{v:.0f}%" for v in dfm['Rate']],
                textposition="top center",
                line=dict(color='#10b981',width=3,shape='spline'),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=dfm['M'], y=[85]*len(dfm),
                mode='lines',
                line=dict(color='#ef4444',dash='dash'),
                name='Meta 85%'
            ))
            fig.update_layout(
                title="Sell-Through Rate Mês a Mês",
                yaxis=dict(range=[0,150],ticksuffix='%'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with a2:
            dfm['SellInMi']  = dfm['Valor_Total']       / 1e6
            dfm['SellOutMi'] = dfm['Valor_SellThrough'] / 1e6

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dfm['M'], y=dfm['SellInMi'],
                mode='lines+markers+text',
                text=[f"{v:.1f} Mi" for v in dfm['SellInMi']],
                textposition='top center',
                line=dict(color='#8884d8',width=3,shape='spline'),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=dfm['M'], y=dfm['SellOutMi'],
                mode='lines+markers+text',
                text=[f"{v:.1f} Mi" for v in dfm['SellOutMi']],
                textposition='bottom center',
                line=dict(color='#82ca9d',width=3,shape='spline'),
                marker=dict(size=6)
            ))
            fig.update_layout(
                title="Sell In vs Sell Out (Milhões de R$)",
                xaxis=dict(tickangle=45,title="Mês/Ano"),
                yaxis=dict(title="Milhões de R$",
                           tickformat=".1f",ticksuffix=" Mi"),
                hovermode='x unified',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Roscas
        st.markdown("---")
        r1, r2 = st.columns(2)
        with r1:
            top5 = sell_in.groupby('Tipo_Produto')['Valor_Total'].sum().nlargest(5).reset_index()
            fig = px.pie(top5, names='Tipo_Produto', values='Valor_Total', hole=0.6,
                         title="Share Tipo Produto", template="plotly_white")
            fig.update_traces(textinfo="percent")
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            byreg = sell_in.groupby('Regiao')['Valor_Total'].sum().reset_index()
            fig = px.pie(byreg, names='Regiao', values='Valor_Total', hole=0.6,
                         title="Distribuição por Região", template="plotly_white")
            fig.update_traces(textinfo="percent")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    if sell_in is None or sell_out is None:
        st.warning("⚠️ Faça upload de ambos os arquivos para ver o Forecast.")
    else:
        st.markdown("### 🔮 Forecast Ajustado de Sell In com base no Sell Out")

        # 1) Série histórica em milhões
        mi = sell_in.groupby('Month')['Valor_Total'].sum() / 1e6
        mo = sell_out.groupby('Month')['Valor_SellThrough'].sum() / 1e6
        dfm = pd.DataFrame({
            'Month': mi.index,
            'SellInMi': mi.values,
            'SellOutMi': mo.reindex(mi.index, fill_value=0).values
        })
        dfm['Rate'] = (dfm['SellOutMi'] / dfm['SellInMi'] * 100).clip(0,150)
        dfm = dfm[dfm['Month'] >= '2023-01'].reset_index(drop=True)
        dfm['M'] = dfm['Month'].apply(format_month)

        # 2) Taxa média de Sell-Through (decimal)
        avg_rate = (dfm['SellOutMi'] / dfm['SellInMi']).mean()

        # 3) Ajusta regressão linear ao Sell Out
        X = np.arange(len(dfm))
        slope, intercept = np.polyfit(X, dfm['SellOutMi'], 1)

        # 4) Projeta próximos 3 meses
        future_X = np.arange(len(dfm), len(dfm) + 3)
        proj_out = intercept + slope * future_X
        proj_out = np.clip(proj_out, a_min=0, a_max=None)
        proj_in  = proj_out / avg_rate

        # 5) Gera labels dos próximos meses
        last = pd.to_datetime(dfm['Month'].iat[-1] + '-01')
        future_months = [
            (last + pd.DateOffset(months=i+1)).strftime('%Y-%m')
            for i in range(3)
        ]
        future_M = [format_month(m) for m in future_months]

        # 6) Concatena histórico + forecast para o gráfico
        all_M = dfm['M'].tolist() + future_M
        all_out = dfm['SellOutMi'].tolist() + proj_out.tolist()
        all_in  = dfm['SellInMi'].tolist()  + proj_in.tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=all_M, y=all_out,
            mode='lines+markers+text',
            name='Sell Out',
            line=dict(color='#82ca9d', shape='spline', width=3),
            text=[f"{v:.1f} Mi" for v in all_out],
            textposition='top center',
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=all_M, y=all_in,
            mode='lines+markers+text',
            name='Sell In Ajustado',
            line=dict(color='#8884d8', shape='spline', width=3, dash='dash'),
            text=[f"{v:.1f} Mi" for v in all_in],
            textposition='bottom center',
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Forecast de Sell Out e Sell In Ajustado (Mi R$)",
            xaxis=dict(title="Mês/Ano", tickangle=45),
            yaxis=dict(title="Milhões de R$", tickformat=".1f", ticksuffix=" Mi"),
            hovermode='x unified',
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # 7) Cards métricos para cada projeção
        st.markdown("### 📊 Projeção para os próximos 3 meses")
        for m, out_mi, in_mi in zip(future_M, proj_out, proj_in):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{m} – Sell Out", f"{out_mi:.1f} Mi")
            with col2:
                st.metric(f"{m} – Sell In Ajustado", f"{in_mi:.1f} Mi")


with tab3:
    st.info("💡 Em breve as Recomendações Estratégicas.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>Desenvolvido com Streamlit + Plotly</p>",
    unsafe_allow_html=True
)
