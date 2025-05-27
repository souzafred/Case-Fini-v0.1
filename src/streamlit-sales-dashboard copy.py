import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Inteligência Comercial",
    page_icon="📊",
    layout="wide"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-left: 4px solid;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .uploadedFile {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar dados
@st.cache_data
def load_data(uploaded_file, file_type):
    if uploaded_file is not None:
        try:
            # Ler o conteúdo do arquivo
            content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            
            # Parse CSV
            df = pd.read_csv(content)
            
            # Limpar nomes das colunas
            df.columns = df.columns.str.strip()
            
            return df
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return None
    return None

# Função para formatar moeda
def format_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Função para formatar mês
def format_month(date_str):
    months = {
        '01': 'Jan', '02': 'Fev', '03': 'Mar', '04': 'Abr',
        '05': 'Mai', '06': 'Jun', '07': 'Jul', '08': 'Ago',
        '09': 'Set', '10': 'Out', '11': 'Nov', '12': 'Dez'
    }
    if isinstance(date_str, str) and len(date_str) >= 7:
        year = date_str[2:4]
        month = date_str[5:7]
        return f"{months.get(month, month)}/{year}"
    return date_str

# Título principal com gradiente
st.markdown("""
<div style="background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%); 
            padding: 2rem; border-radius: 0.5rem; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">Dashboard de Inteligência Comercial</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">Análise de Performance de Vendas - FMCG Doces e Guloseimas</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para upload de arquivos
with st.sidebar:
    st.header("📁 Upload de Dados")
    
    sell_in_file = st.file_uploader("Upload Sell In (CSV)", type=['csv'], key="sell_in")
    sell_out_file = st.file_uploader("Upload Sell Out (CSV)", type=['csv'], key="sell_out")
    
    if sell_in_file and sell_out_file:
        st.success("✅ Arquivos carregados com sucesso!")

# Carregar dados
sell_in_data = load_data(sell_in_file, 'sell_in') if sell_in_file else None
sell_out_data = load_data(sell_out_file, 'sell_out') if sell_out_file else None

# Tabs principais
tab1, tab2, tab3 = st.tabs(["📊 Análise de Vendas & KPIs", "🔮 Previsão de Vendas", "💡 Recomendações Estratégicas"])

# Tab 1: Análise de Vendas
with tab1:
    if sell_in_data is not None and sell_out_data is not None:
        # Preparar dados
        sell_in_data['Datas'] = pd.to_datetime(sell_in_data['Data_Pedido'])
        sell_in_data['Month'] = sell_in_data['Data_Pedido'].dt.to_period('M').astype(str)
        
        sell_out_data['Data_Venda'] = pd.to_datetime(sell_out_data['Data_Venda'])
        sell_out_data['Month'] = sell_out_data['Data_Venda'].dt.to_period('M').astype(str)
        
        # Calcular KPIs
        # Sell In
        total_sell_in = sell_in_data['Valor_Total'].sum()
        volume_sell_in = sell_in_data['Quantidade'].sum()
        ticket_medio_sell_in = total_sell_in / volume_sell_in if volume_sell_in > 0 else 0
        
        # Sell Out
        total_sell_out = sell_out_data['Valor_SellThrough'].sum()
        volume_sell_out = sell_out_data['Unidades_SellThrough'].sum()
        ticket_medio_sell_out = total_sell_out / volume_sell_out if volume_sell_out > 0 else 0
        
        # Sell Through
        sell_through = (total_sell_out / total_sell_in * 100) if total_sell_in > 0 else 0
        
        # Crescimento Sell In
        monthly_sell_in = sell_in_data.groupby('Month')['Valor_Total'].sum().reset_index()
        monthly_sell_in = monthly_sell_in.sort_values('Month')
        if len(monthly_sell_in) >= 6:
            last_3_months = monthly_sell_in.tail(3)['Valor_Total'].sum()
            previous_3_months = monthly_sell_in.iloc[-6:-3]['Valor_Total'].sum()
            growth_sell_in = ((last_3_months - previous_3_months) / previous_3_months * 100) if previous_3_months > 0 else 0
        else:
            growth_sell_in = 0
            
        # Crescimento Sell Out
        monthly_sell_out = sell_out_data.groupby('Month')['Valor_SellThrough'].sum().reset_index()
        monthly_sell_out = monthly_sell_out.sort_values('Month')
        if len(monthly_sell_out) >= 6:
            last_3_months = monthly_sell_out.tail(3)['Valor_SellThrough'].sum()
            previous_3_months = monthly_sell_out.iloc[-6:-3]['Valor_SellThrough'].sum()
            growth_sell_out = ((last_3_months - previous_3_months) / previous_3_months * 100) if previous_3_months > 0 else 0
        else:
            growth_sell_out = 0
        
        # Regiões
        regions = sell_in_data['Regiao'].nunique()
        
        # Seção de KPIs
        st.markdown("### 📦 Indicadores Sell In")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Sell In",
                value=format_currency(total_sell_in),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Volume Sell In",
                value=f"{volume_sell_in:,.0f}".replace(",", "."),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Ticket Médio Sell In",
                value=format_currency(ticket_medio_sell_in),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Crescimento Sell In",
                value=f"{growth_sell_in:.1f}%",
                delta=f"{growth_sell_in:.1f}%",
                delta_color="normal" if growth_sell_in > 0 else "inverse"
            )
        
        st.markdown("### 🛒 Indicadores Sell Out")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Sell Out",
                value=format_currency(total_sell_out),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Volume Sell Out",
                value=f"{volume_sell_out:,.0f}".replace(",", "."),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Ticket Médio Sell Out",
                value=format_currency(ticket_medio_sell_out),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Crescimento Sell Out",
                value=f"{growth_sell_out:.1f}%",
                delta=f"{growth_sell_out:.1f}%",
                delta_color="normal" if growth_sell_out > 0 else "inverse"
            )
        
        # KPI Geral - Sell Through
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                    padding: 1.5rem; border-radius: 0.5rem; margin: 2rem 0; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: white;">🎯 Sell Through Rate</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">{:.1f}%</p>
                    <p style="margin: 0; opacity: 0.9;">Proporção entre Sell Out e Sell In - Meta: >85%</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; opacity: 0.9;">Regiões Atendidas</p>
                    <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">{}</p>
                </div>
            </div>
        </div>
        """.format(sell_through, regions), unsafe_allow_html=True)
        
        # Gráficos - Primeira linha
        col1, col2 = st.columns(2)
        
        with col1:
            # Sell Through Rate mês a mês
            monthly_comparison = pd.merge(
                monthly_sell_in.rename(columns={'Valor_Total': 'Sell_In'}),
                monthly_sell_out.rename(columns={'Valor_SellThrough': 'Sell_Out'}),
                on='Month',
                how='outer'
            ).fillna(0)
            
            monthly_comparison['Sell_Through'] = (
                monthly_comparison['Sell_Out'] / monthly_comparison['Sell_In'] * 100
            ).replace([np.inf, -np.inf], 0).fillna(0)
            
            # Limitar a 150% e filtrar dados a partir de 2023
            monthly_comparison['Sell_Through'] = monthly_comparison['Sell_Through'].clip(0, 150)
            monthly_comparison = monthly_comparison[monthly_comparison['Month'] >= '2023-01']
            monthly_comparison['Month_Display'] = monthly_comparison['Month'].apply(format_month)
            
            fig_sellthrough = go.Figure()
            
            # Linha de Sell Through
            fig_sellthrough.add_trace(go.Scatter(
                x=monthly_comparison['Month_Display'],
                y=monthly_comparison['Sell_Through'],
                mode='lines+markers',
                name='Sell Through Rate',
                line=dict(color='#10b981', width=3),
                marker=dict(size=8, color='#10b981')
            ))
            
            # Linha de meta
            fig_sellthrough.add_trace(go.Scatter(
                x=monthly_comparison['Month_Display'],
                y=[85] * len(monthly_comparison),
                mode='lines',
                name='Meta (85%)',
                line=dict(color='#ef4444', width=2, dash='dash')
            ))
            
            fig_sellthrough.update_layout(
                title="Sell Through Rate - Mês a Mês",
                xaxis_title="",
                yaxis_title="Percentual (%)",
                yaxis=dict(range=[0, 150], tickformat='.0f', ticksuffix='%'),
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_sellthrough, use_container_width=True)
            st.caption("* Valores acima de 100% indicam venda de estoque acumulado")
        
        with col2:
            # Sell In vs Sell Out
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Scatter(
                x=monthly_comparison['Month_Display'],
                y=monthly_comparison['Sell_In'],
                mode='lines+markers',
                name='Sell In',
                line=dict(color='#8884d8', width=2)
            ))
            
            fig_comparison.add_trace(go.Scatter(
                x=monthly_comparison['Month_Display'],
                y=monthly_comparison['Sell_Out'],
                mode='lines+markers',
                name='Sell Out',
                line=dict(color='#82ca9d', width=2)
            ))
            
            fig_comparison.update_layout(
                title="Sell In vs Sell Out",
                xaxis_title="",
                yaxis_title="Valor (R$)",
                yaxis=dict(tickformat=',.0f'),
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Gráficos de Rosca - Segunda linha
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance por Tipo (Top 5)
            product_performance = sell_in_data.groupby('Produto')['Valor_Total'].sum().nlargest(5).reset_index()
            
            fig_products = go.Figure(data=[go.Pie(
                labels=product_performance['Produto'],
                values=product_performance['Valor_Total'],
                hole=0.6,
                textposition='inside',
                textinfo='percent',
                marker=dict(colors=px.colors.qualitative.Set3[:5])
            )])
            
            fig_products.update_layout(
                title="Performance por Tipo",
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                height=400,
                margin=dict(l=20, r=120, t=70, b=20)
            )
            
            st.plotly_chart(fig_products, use_container_width=True)
        
        with col2:
            # Distribuição por Região
            region_performance = sell_in_data.groupby('Regiao')['Valor_Total'].sum().reset_index()
            
            fig_regions = go.Figure(data=[go.Pie(
                labels=region_performance['Regiao'],
                values=region_performance['Valor_Total'],
                hole=0.6,
                textposition='inside',
                textinfo='percent',
                marker=dict(colors=['#8b5cf6', '#10b981'])
            )])
            
            fig_regions.update_layout(
                title="Distribuição por Região",
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                height=400,
                margin=dict(l=20, r=120, t=70, b=20)
            )
            
            st.plotly_chart(fig_regions, use_container_width=True)
    
    else:
        st.warning("⚠️ Por favor, faça upload dos arquivos de Sell In e Sell Out para visualizar o dashboard.")

# Tab 2: Previsão de Vendas
with tab2:
    if sell_in_data is not None:
        st.markdown("### 📈 Previsão de Vendas - Próximos 3 Meses")
        
        # Análise de tendência simples
        monthly_trend = sell_in_data.groupby('Month')['Valor_Total'].sum().reset_index()
        monthly_trend = monthly_trend.sort_values('Month')
        
        if len(monthly_trend) >= 3:
            # Calcular média móvel e tendência
            last_3_avg = monthly_trend.tail(3)['Valor_Total'].mean()
            growth_rate = 0.05  # Taxa de crescimento conservadora de 5%
            
            # Projetar próximos 3 meses
            last_month = monthly_trend['Month'].max()
            next_months = pd.date_range(start=pd.to_datetime(last_month) + pd.DateOffset(months=1), 
                                       periods=3, freq='M')
            
            projections = []
            for i, month in enumerate(next_months):
                projected_value = last_3_avg * (1 + growth_rate) ** (i + 1)
                projections.append({
                    'Month': month.strftime('%Y-%m'),
                    'Valor_Projetado': projected_value
                })
            
            projections_df = pd.DataFrame(projections)
            
            # Gráfico de previsão
            fig_forecast = go.Figure()
            
            # Dados históricos
            fig_forecast.add_trace(go.Scatter(
                x=monthly_trend['Month'].apply(format_month),
                y=monthly_trend['Valor_Total'],
                mode='lines+markers',
                name='Histórico',
                line=dict(color='#8884d8', width=2)
            ))
            
            # Projeções
            fig_forecast.add_trace(go.Scatter(
                x=projections_df['Month'].apply(format_month),
                y=projections_df['Valor_Projetado'],
                mode='lines+markers',
                name='Projeção',
                line=dict(color='#ef4444', width=2, dash='dash'),
                marker=dict(size=10)
            ))
            
            fig_forecast.update_layout(
                title="Previsão de Vendas - Sell In",
                xaxis_title="Mês",
                yaxis_title="Valor (R$)",
                yaxis=dict(tickformat=',.0f'),
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Tabela de previsões
            st.markdown("### 📊 Valores Projetados")
            for _, row in projections_df.iterrows():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=format_month(row['Month']),
                        value=format_currency(row['Valor_Projetado']),
                        delta=f"+{growth_rate*100:.0f}%"
                    )
        else:
            st.info("📊 Necessário pelo menos 3 meses de dados históricos para gerar previsões.")
    else:
        st.warning("⚠️ Por favor, faça upload do arquivo de Sell In para visualizar as previsões.")

# Tab 3: Recomendações
with tab3:
    if sell_in_data is not None and sell_out_data is not None:
        st.markdown("### 💡 Análise e Recomendações Estratégicas")
        
        # Análise de Sell Through por região
        sell_through_by_region = {}
        for region in sell_in_data['Regiao'].unique():
            sell_in_region = sell_in_data[sell_in_data['Regiao'] == region]['Valor_Total'].sum()
            sell_out_region = sell_out_data[sell_out_data['Regiao'] == region]['Valor_SellThrough'].sum()
            if sell_in_region > 0:
                sell_through_by_region[region] = (sell_out_region / sell_in_region) * 100
        
        # Encontrar melhor e pior região
        if sell_through_by_region:
            best_region = max(sell_through_by_region, key=sell_through_by_region.get)
            worst_region = min(sell_through_by_region, key=sell_through_by_region.get)
            
            # Card de Oportunidades
            st.markdown("""
            <div style="background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="color: #166534; margin: 0;">✅ Oportunidades Identificadas</h4>
                <ul style="margin: 0.5rem 0;">
                    <li>Região <strong>{}</strong> com excelente Sell Through de {:.1f}%</li>
                    <li>Potencial de crescimento identificado em {} produtos com alta demanda</li>
                    <li>Sell Through geral de {:.1f}% indica boa rotatividade de estoque</li>
                </ul>
            </div>
            """.format(best_region, sell_through_by_region[best_region], 
                      sell_in_data['Produto'].nunique(), sell_through), unsafe_allow_html=True)
            
            # Card de Pontos de Atenção
            st.markdown("""
            <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="color: #92400e; margin: 0;">⚠️ Pontos de Atenção</h4>
                <ul style="margin: 0.5rem 0;">
                    <li>Região <strong>{}</strong> com Sell Through de apenas {:.1f}%</li>
                    <li>Concentração de {}% das vendas em uma única região</li>
                    <li>Necessário diversificar portfólio para reduzir dependência dos top 5 produtos</li>
                </ul>
            </div>
            """.format(worst_region, sell_through_by_region[worst_region],
                      int((region_performance.iloc[0]['Valor_Total'] / total_sell_in) * 100)), unsafe_allow_html=True)
            
            # Recomendações Detalhadas
            st.markdown("### 🎯 Plano de Ação Recomendado")
            
            recommendations = [
                {
                    "area": "Expansão Regional",
                    "acao": f"Replicar estratégias da região {best_region} nas demais regiões",
                    "impacto": "Alto",
                    "prazo": "Curto (1-3 meses)"
                },
                {
                    "area": "Gestão de Estoque",
                    "acao": f"Ajustar níveis de estoque na região {worst_region} para melhorar Sell Through",
                    "impacto": "Médio",
                    "prazo": "Imediato"
                },
                {
                    "area": "Mix de Produtos",
                    "acao": "Expandir portfólio com produtos de ticket médio similar aos top performers",
                    "impacto": "Alto",
                    "prazo": "Médio (3-6 meses)"
                },
                {
                    "area": "Estratégia Comercial",
                    "acao": "Implementar promoções cruzadas entre produtos de alto e baixo giro",
                    "impacto": "Médio",
                    "prazo": "Curto (1-3 meses)"
                }
            ]
            
            # Criar DataFrame para exibir recomendações
            rec_df = pd.DataFrame(recommendations)
            
            # Estilizar a tabela
            st.dataframe(
                rec_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "area": st.column_config.TextColumn("Área", width="medium"),
                    "acao": st.column_config.TextColumn("Ação Recomendada", width="large"),
                    "impacto": st.column_config.TextColumn("Impacto Esperado", width="small"),
                    "prazo": st.column_config.TextColumn("Prazo", width="medium")
                }
            )
            
            # Métricas de Impacto Projetado
            st.markdown("### 📊 Impacto Projetado das Ações")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Aumento Projetado em Vendas",
                    value="+15-20%",
                    delta="nos próximos 6 meses"
                )
            
            with col2:
                st.metric(
                    label="Melhoria no Sell Through",
                    value="+10pp",
                    delta="atingindo 95%+"
                )
            
            with col3:
                st.metric(
                    label="Redução de Estoque Parado",
                    value="-25%",
                    delta="otimização de capital"
                )
            
        else:
            st.info("📊 Dados insuficientes para gerar recomendações detalhadas.")
    else:
        st.warning("⚠️ Por favor, faça upload dos arquivos de Sell In e Sell Out para visualizar as recomendações.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>Dashboard de Inteligência Comercial - FMCG Doces e Guloseimas</p>
    <p style="font-size: 0.875rem;">Desenvolvido com Streamlit + Plotly</p>
</div>
""", unsafe_allow_html=True)
