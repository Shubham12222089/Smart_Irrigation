import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Smart Irrigation Dashboard", layout="wide")

# Load data
df = pd.read_csv("predictions.csv")

# Load model metrics if available
try:
    metrics_df = pd.read_csv("model_metrics.csv")
    has_metrics = True
except:
    has_metrics = False

# Load all model predictions if available
try:
    all_models_df = pd.read_csv("all_model_predictions.csv")
    has_all_models = True
except:
    has_all_models = False
    all_models_df = df

st.title("Smart Irrigation System Dashboard")
st.markdown("### Multi-Model AI-Powered Irrigation Insights")

# Add summary metrics at the top
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    avg_moisture = df['Moisture'].mean()
    st.metric("Avg Moisture", f"{avg_moisture:.1f}%")
with col3:
    st.metric("Crop Types", df['Crop Type'].nunique())
with col4:
    st.metric("Soil Types", df['Soil Type'].nunique())

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Model Performance",
    "Model Comparison",
    "Crop Insights",
    "Soil Insights",
    "Environmental Analysis",
    "Irrigation Recommendation"
])

# ---------- TAB 1: MODEL PERFORMANCE ----------
with tab1:
    st.header("Linear Regression Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Moisture")
        fig1 = px.scatter(df, x="Moisture", y="prediction", 
                         title="Actual vs Predicted Moisture Levels",
                         labels={"Moisture": "Actual Moisture (%)", "prediction": "Predicted Moisture (%)"},
                         color_discrete_sequence=["#FF6B6B"],
                         trendline="ols")
        fig1.add_trace(go.Scatter(x=[df['Moisture'].min(), df['Moisture'].max()],
                                  y=[df['Moisture'].min(), df['Moisture'].max()],
                                  mode='lines', name='Perfect Prediction',
                                  line=dict(color='black', dash='dash', width=2)))
        st.plotly_chart(fig1, use_container_width=True)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(df['Moisture'], df['prediction']))
        r2 = r2_score(df['Moisture'], df['prediction'])
        mae = mean_absolute_error(df['Moisture'], df['prediction'])
        
        st.markdown(f"""
        **Model Metrics:**
        - **R² Score**: {r2:.4f} ({r2*100:.2f}% accuracy)
        - **RMSE**: {rmse:.4f}
        - **MAE**: {mae:.4f}
        """)
    
    with col2:
        st.subheader("Prediction Error Distribution")
        df['error'] = df['Moisture'] - df['prediction']
        fig2 = px.histogram(df, x="error", nbins=50,
                           title="Distribution of Prediction Errors",
                           labels={"error": "Prediction Error (%)"},
                           color_discrete_sequence=["#4ECDC4"])
        fig2.add_vline(x=0, line_dash="dash", line_color="red", 
                      annotation_text="Zero Error", annotation_position="top")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown(f"""
        **Error Statistics:**
        - **Mean Error**: {df['error'].mean():.4f}
        - **Std Dev**: {df['error'].std():.4f}
        - **Min Error**: {df['error'].min():.4f}
        - **Max Error**: {df['error'].max():.4f}
        """)
    
    # Residual plot
    st.subheader("Residual Analysis")
    fig3 = px.scatter(df, x="prediction", y="error",
                     title="Residual Plot: Predicted vs Error",
                     labels={"prediction": "Predicted Moisture (%)", "error": "Residual Error"},
                     color_discrete_sequence=["#45B7D1"])
    fig3.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Distribution comparison
    st.subheader("Distribution Comparison")
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=df['Moisture'], name='Actual', opacity=0.7, marker_color='#FF6B6B'))
    fig4.add_trace(go.Histogram(x=df['prediction'], name='Predicted', opacity=0.7, marker_color='#4ECDC4'))
    fig4.update_layout(barmode='overlay', title="Actual vs Predicted Moisture Distribution",
                      xaxis_title="Moisture (%)", yaxis_title="Count")
    st.plotly_chart(fig4, use_container_width=True)

# ---------- TAB 2: MODEL COMPARISON ----------
with tab2:
    st.header("Multi-Model Comparison")
    
    if has_metrics:
        st.subheader("Model Performance Metrics")
        
        # Display metrics table
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy_R2', 'F1_Score'], 
                                                     color='lightgreen'), use_container_width=True)
        
        # Model accuracy comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            fig_acc = px.bar(metrics_df, x='Model', y='Accuracy_R2',
                           title="Model Accuracy Comparison",
                           labels={"Accuracy_R2": "Accuracy (%)"},
                           color='Model',
                           color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#45B7D1"])
            fig_acc.update_traces(text=metrics_df['Accuracy_R2'].round(2), textposition='outside')
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            st.subheader("F1 Score Comparison (Classification Models)")
            classification_metrics = metrics_df[metrics_df['Type'] == 'Classification']
            fig_f1 = px.bar(classification_metrics, x='Model', y='F1_Score',
                          title="F1 Score for Classification Models",
                          color='Model',
                          color_discrete_sequence=["#4ECDC4", "#45B7D1"])
            fig_f1.update_traces(text=classification_metrics['F1_Score'].round(4), textposition='outside')
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Radar chart for classification models
        st.subheader("Classification Models: Multi-Metric Comparison")
        classification_metrics = metrics_df[metrics_df['Type'] == 'Classification'].fillna(0)
        
        fig_radar = go.Figure()
        
        for idx, row in classification_metrics.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy_R2'], row['F1_Score']*100, row['Precision']*100, row['Recall']*100],
                theta=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Multi-Metric Performance Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    else:
        st.info("Model metrics file not found. Run the model training script first to generate model_metrics.csv")
    
    if has_all_models:
        st.subheader("Prediction Comparison Across Models")
        
        # Show sample predictions
        sample_size = min(20, len(all_models_df))
        sample_df = all_models_df.head(sample_size)
        
        st.dataframe(sample_df[['Moisture', 'LR_Prediction', 'Logistic_Prediction', 'DecisionTree_Prediction']], 
                    use_container_width=True)
        
        # Category distribution comparison
        st.subheader("Moisture Category Distribution by Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_log_dist = px.pie(all_models_df, names='Logistic_Prediction',
                                 title="Logistic Regression Categories",
                                 color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig_log_dist, use_container_width=True)
        
        with col2:
            fig_dt_dist = px.pie(all_models_df, names='DecisionTree_Prediction',
                                 title="Decision Tree Categories",
                                 color_discrete_sequence=px.colors.sequential.Blues)
            st.plotly_chart(fig_dt_dist, use_container_width=True)
    else:
        st.info("Multi-model predictions file not found. Run the enhanced model training script to compare all models.")


# ---------- TAB 3: CROP INSIGHTS ----------
with tab3:
    st.header("Crop-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Moisture by Crop Type")
        crop_moisture = df.groupby("Crop Type")["Moisture"].mean().reset_index().sort_values("Moisture", ascending=False)
        fig3 = px.bar(crop_moisture, x="Crop Type", y="Moisture",
                     title="Average Moisture Levels by Crop",
                     color="Moisture",
                     color_continuous_scale="Greens")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("Crop Type Distribution")
        crop_counts = df['Crop Type'].value_counts().reset_index()
        crop_counts.columns = ['Crop Type', 'Count']
        fig_crop_dist = px.pie(crop_counts, values='Count', names='Crop Type',
                              title="Distribution of Crop Types",
                              color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_crop_dist, use_container_width=True)

    # Interactive crop selector
    st.subheader("Detailed Crop Analysis")
    crop = st.selectbox("Select Crop Type", df["Crop Type"].unique())
    crop_data = df[df["Crop Type"] == crop]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", len(crop_data))
    with col2:
        st.metric("Avg Moisture", f"{crop_data['Moisture'].mean():.2f}%")
    with col3:
        st.metric("Moisture Range", f"{crop_data['Moisture'].min():.1f} - {crop_data['Moisture'].max():.1f}")
    
    # Moisture trend for selected crop
    fig_crop_trend = go.Figure()
    fig_crop_trend.add_trace(go.Scatter(y=crop_data['Moisture'].values, mode='lines+markers',
                                       name='Actual', line=dict(color='#FF6B6B', width=2)))
    fig_crop_trend.add_trace(go.Scatter(y=crop_data['prediction'].values, mode='lines+markers',
                                       name='Predicted', line=dict(color='#4ECDC4', width=2)))
    fig_crop_trend.update_layout(title=f"Moisture Trend for {crop}",
                                xaxis_title="Sample Index", yaxis_title="Moisture (%)")
    st.plotly_chart(fig_crop_trend, use_container_width=True)
    
    # Box plot for all crops
    st.subheader("Moisture Distribution Across All Crops")
    fig_box = px.box(df, x="Crop Type", y="Moisture", color="Crop Type",
                    title="Moisture Variance by Crop Type")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Crop vs Environmental factors
    st.subheader("Crop Environmental Requirements")
    fig_env = make_subplots(rows=1, cols=3, 
                           subplot_titles=("Temperature", "Humidity", "Moisture"))
    
    for crop_type in df['Crop Type'].unique():
        crop_subset = df[df['Crop Type'] == crop_type]
        fig_env.add_trace(go.Box(y=crop_subset['Temparature'], name=crop_type, showlegend=False), row=1, col=1)
        fig_env.add_trace(go.Box(y=crop_subset['Humidity'], name=crop_type, showlegend=False), row=1, col=2)
        fig_env.add_trace(go.Box(y=crop_subset['Moisture'], name=crop_type), row=1, col=3)
    
    fig_env.update_layout(height=400, title_text="Environmental Factors by Crop Type")
    st.plotly_chart(fig_env, use_container_width=True)


# ---------- TAB 4: SOIL INSIGHTS ----------
with tab4:
    st.header("Soil-Based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Moisture by Soil Type")
        soil_moisture = df.groupby("Soil Type")["Moisture"].mean().reset_index().sort_values("Moisture", ascending=False)
        fig4 = px.bar(soil_moisture, x="Soil Type", y="Moisture",
                     title="Average Moisture Levels by Soil",
                     color="Soil Type",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.subheader("Soil Type Distribution")
        soil_counts = df['Soil Type'].value_counts().reset_index()
        soil_counts.columns = ['Soil Type', 'Count']
        fig_soil_dist = px.pie(soil_counts, values='Count', names='Soil Type',
                              title="Distribution of Soil Types",
                              color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig_soil_dist, use_container_width=True)

    # Interactive soil selector
    st.subheader("Detailed Soil Analysis")
    soil = st.selectbox("Select Soil Type", df["Soil Type"].unique())
    soil_data = df[df["Soil Type"] == soil]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", len(soil_data))
    with col2:
        st.metric("Avg Moisture", f"{soil_data['Moisture'].mean():.2f}%")
    with col3:
        st.metric("Moisture Range", f"{soil_data['Moisture'].min():.1f} - {soil_data['Moisture'].max():.1f}")
    
    # Moisture trend for selected soil
    fig_soil_trend = go.Figure()
    fig_soil_trend.add_trace(go.Scatter(y=soil_data['Moisture'].values, mode='lines+markers',
                                       name='Actual', line=dict(color='#FF6B6B', width=2)))
    fig_soil_trend.add_trace(go.Scatter(y=soil_data['prediction'].values, mode='lines+markers',
                                       name='Predicted', line=dict(color='#4ECDC4', width=2)))
    fig_soil_trend.update_layout(title=f"Moisture Trend for {soil}",
                                xaxis_title="Sample Index", yaxis_title="Moisture (%)")
    st.plotly_chart(fig_soil_trend, use_container_width=True)
    
    # Violin plot for soil types
    st.subheader("Moisture Distribution Pattern by Soil Type")
    fig_violin = px.violin(df, x="Soil Type", y="Moisture", color="Soil Type", box=True,
                          title="Detailed Moisture Distribution by Soil Type")
    st.plotly_chart(fig_violin, use_container_width=True)
    
    # Soil nutrients analysis
    st.subheader("Nutrient Levels by Soil Type")
    nutrient_cols = ['Nitrogen', 'Potassium', 'Phosphorous']
    
    fig_nutrients = make_subplots(rows=1, cols=3, 
                                 subplot_titles=("Nitrogen", "Potassium", "Phosphorous"))
    
    for soil_type in df['Soil Type'].unique():
        soil_subset = df[df['Soil Type'] == soil_type]
        fig_nutrients.add_trace(go.Box(y=soil_subset['Nitrogen'], name=soil_type, showlegend=False), row=1, col=1)
        fig_nutrients.add_trace(go.Box(y=soil_subset['Potassium'], name=soil_type, showlegend=False), row=1, col=2)
        fig_nutrients.add_trace(go.Box(y=soil_subset['Phosphorous'], name=soil_type), row=1, col=3)
    
    fig_nutrients.update_layout(height=400, title_text="Nutrient Distribution by Soil Type")
    st.plotly_chart(fig_nutrients, use_container_width=True)


# ---------- TAB 5: ENVIRONMENTAL ANALYSIS ----------
with tab5:
    st.header("Environmental Factors Analysis")
    
    # Temperature vs Humidity scatter
    st.subheader("Temperature vs Humidity Relationship")
    fig_temp_hum = px.scatter(df, x="Temparature", y="Humidity", color="Moisture",
                             size="Moisture", hover_data=["Crop Type", "Soil Type"],
                             title="Temperature vs Humidity (colored by Moisture)",
                             color_continuous_scale="Viridis")
    st.plotly_chart(fig_temp_hum, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Distribution")
        fig_temp = px.histogram(df, x="Temparature", nbins=30,
                               title="Temperature Distribution",
                               color_discrete_sequence=["#FF6B6B"])
        st.plotly_chart(fig_temp, use_container_width=True)
        
        st.metric("Avg Temperature", f"{df['Temparature'].mean():.2f}°C")
    
    with col2:
        st.subheader("Humidity Distribution")
        fig_hum = px.histogram(df, x="Humidity", nbins=30,
                              title="Humidity Distribution",
                              color_discrete_sequence=["#4ECDC4"])
        st.plotly_chart(fig_hum, use_container_width=True)
        
        st.metric("Avg Humidity", f"{df['Humidity'].mean():.2f}%")
    
    # Correlation heatmap
    st.subheader("Feature Correlation Analysis")
    numeric_cols = ['Temparature', 'Humidity', 'Nitrogen', 'Potassium', 'Phosphorous', 'Moisture', 'prediction']
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        text_auto='.2f',
                        aspect="auto",
                        title="Correlation Heatmap of Features",
                        color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("3D Environmental Space")
    fig_3d = px.scatter_3d(df.sample(min(500, len(df))), 
                          x="Temparature", y="Humidity", z="Moisture",
                          color="Crop Type",
                          title="3D View: Temperature, Humidity, and Moisture",
                          opacity=0.7)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # NPK Analysis
    st.subheader("NPK (Nitrogen-Phosphorous-Potassium) Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_n = px.box(df, y="Nitrogen", color="Crop Type",
                      title="Nitrogen Levels by Crop")
        st.plotly_chart(fig_n, use_container_width=True)
    
    with col2:
        fig_p = px.box(df, y="Phosphorous", color="Crop Type",
                      title="Phosphorous Levels by Crop")
        st.plotly_chart(fig_p, use_container_width=True)
    
    with col3:
        fig_k = px.box(df, y="Potassium", color="Crop Type",
                      title="Potassium Levels by Crop")
        st.plotly_chart(fig_k, use_container_width=True)


# ---------- TAB 6: IRRIGATION RECOMMENDATION ----------

with tab6:
    st.header("Smart Irrigation Recommendations")

    def irrigation_level(row):
        m = row["prediction"]
        crop_type = row["Crop Type"]
        soil_type = row["Soil Type"]

        # Adjust thresholds based on crop and soil type
        if crop_type == 'Wheat' and soil_type == 'Loamy':
            if m < 25:
                return "High Water Needed"
            elif m < 45:
                return "Moderate Water Needed"
            else:
                return "Low Water Needed"
        elif crop_type == 'Maize' and soil_type == 'Sandy':
            if m < 35:
                return "High Water Needed"
            elif m < 55:
                return "Moderate Water Needed"
            else:
                return "Low Water Needed"
        # Default case
        else:
            if m < 30:
                return "High Water Needed"
            elif m < 50:
                return "Moderate Water Needed"
            else:
                return "Low Water Needed"

    df["Irrigation_Level"] = df.apply(irrigation_level, axis=1)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    high_water = len(df[df['Irrigation_Level'].str.contains('High')])
    moderate_water = len(df[df['Irrigation_Level'].str.contains('Moderate')])
    low_water = len(df[df['Irrigation_Level'].str.contains('Low')])
    
    with col1:
        st.metric("High Water Needed", high_water, f"{high_water/len(df)*100:.1f}%")
    with col2:
        st.metric("Moderate Water", moderate_water, f"{moderate_water/len(df)*100:.1f}%")
    with col3:
        st.metric("Low Water", low_water, f"{low_water/len(df)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Irrigation Distribution")
        irr_counts = df['Irrigation_Level'].value_counts().reset_index()
        irr_counts.columns = ['Level', 'Count']
        fig5 = px.pie(irr_counts, values='Count', names='Level',
                     title="Overall Irrigation Needs Distribution",
                     color='Level',
                     color_discrete_map={
                         'High Water Needed': '#FF6B6B',
                         'Moderate Water Needed': '#FFD93D',
                         'Low Water Needed': '#6BCF7F'
                     })
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.subheader("Irrigation by Crop Type")
        fig_crop_irr = px.histogram(df, x="Crop Type", color="Irrigation_Level",
                                   title="Irrigation Requirements by Crop",
                                   color_discrete_map={
                                       'High Water Needed': '#FF6B6B',
                                       'Moderate Water Needed': '#FFD93D',
                                       'Low Water Needed': '#6BCF7F'
                                   },
                                   barmode='group')
        st.plotly_chart(fig_crop_irr, use_container_width=True)
    
    # Detailed recommendations table
    st.subheader("Detailed Irrigation Recommendations")
    
    # Filter options
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        crop_filter = st.multiselect("Filter by Crop", options=df['Crop Type'].unique(), 
                                     default=df['Crop Type'].unique())
    with filter_col2:
        soil_filter = st.multiselect("Filter by Soil", options=df['Soil Type'].unique(), 
                                     default=df['Soil Type'].unique())
    with filter_col3:
        irr_filter = st.multiselect("Filter by Irrigation Level", 
                                    options=df['Irrigation_Level'].unique(), 
                                    default=df['Irrigation_Level'].unique())
    
    filtered_df = df[
        (df['Crop Type'].isin(crop_filter)) & 
        (df['Soil Type'].isin(soil_filter)) & 
        (df['Irrigation_Level'].isin(irr_filter))
    ]
    
    st.dataframe(
        filtered_df[['Crop Type', 'Soil Type', 'Moisture', 'prediction', 
                    'Temparature', 'Humidity', 'Irrigation_Level']].style.background_gradient(
            subset=['Moisture', 'prediction'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.download_button(
        label="Download Irrigation Report (CSV)",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='irrigation_recommendations.csv',
        mime='text/csv'
    )
    
    # Irrigation by soil type
    st.subheader("Irrigation Requirements by Soil Type")
    fig_soil_irr = px.sunburst(df, path=['Soil Type', 'Irrigation_Level'],
                               title="Hierarchical View: Soil Type → Irrigation Level",
                               color='Irrigation_Level',
                               color_discrete_map={
                                   'High Water Needed': '#FF6B6B',
                                   'Moderate Water Needed': '#FFD93D',
                                   'Low Water Needed': '#6BCF7F'
                               })
    st.plotly_chart(fig_soil_irr, use_container_width=True)
    
    # Priority irrigation zones
    st.subheader("Priority Irrigation Zones")
    
    high_priority = df[df['Irrigation_Level'].str.contains('High')].groupby(['Crop Type', 'Soil Type']).size().reset_index(name='Count').sort_values('Count', ascending=False)
    
    if len(high_priority) > 0:
        st.warning(f"{len(high_priority)} crop-soil combinations require immediate irrigation attention!")
        
        fig_priority = px.bar(high_priority.head(10), x='Count', y='Crop Type',
                            color='Soil Type', orientation='h',
                            title="Top 10 Priority Zones Requiring High Water",
                            labels={'Count': 'Number of Plots'})
        st.plotly_chart(fig_priority, use_container_width=True)
    else:
        st.success("No critical irrigation zones detected!")
    
    # Moisture prediction quality by irrigation level
    st.subheader("Prediction Accuracy by Irrigation Level")
    df['pred_error'] = abs(df['Moisture'] - df['prediction'])
    
    fig_error = px.box(df, x='Irrigation_Level', y='pred_error',
                      title="Prediction Error Distribution by Irrigation Level",
                      color='Irrigation_Level',
                      color_discrete_map={
                          'High Water Needed': '#FF6B6B',
                          'Moderate Water Needed': '#FFD93D',
                          'Low Water Needed': '#6BCF7F'
                      })
    st.plotly_chart(fig_error, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Smart Irrigation System Dashboard v2.0 | Powered by ML Models (Linear Regression, Logistic Regression, Decision Tree)</p>
        <p>Making agriculture smarter, one drop at a time</p>
    </div>
""", unsafe_allow_html=True)
