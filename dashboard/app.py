"""
Global Foundries Wafer Manufacturing Optimization Dashboard
Main Streamlit application for analyzing WM-811K wafer dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="Global Foundries - Wafer Manufacturing Optimization",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🔬 Global Foundries Manufacturing Optimization Dashboard</h1>
        <p style="color: #e8f4fd; margin: 0;">WM-811K Wafer Dataset Analysis & Process Optimization</p>
        <p style="color: #b8d4fd; margin: 0; font-size: 0.9em;">📍 Live Portfolio Demo | 🚀 Production-Ready MLOps Pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["Manufacturing Overview", "Defect Analysis", "Yield Optimization", "Quality Control", "MLOps Pipeline"]
        )
        
        st.divider()
        
        # Data filters (placeholder for now)
        st.header("📊 Data Filters")
        
        # Date range selector
        st.date_input("Production Date Range", value=[])
        
        # Lot selection
        st.multiselect("Manufacturing Lots", options=["Lot_001", "Lot_002", "Lot_003"], default=[])
        
        # Defect type filter
        st.multiselect(
            "Defect Types", 
            options=["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"],
            default=[]
        )
        
        st.divider()
        
        # Dataset info
        st.header("📋 Dataset Info")
        st.info("""
        **WM-811K Dataset**
        - 811,457 wafer maps
        - 8 defect pattern classes
        - 46,293 manufacturing lots
        - Real production data
        """)
    
    # Main content area based on selected page
    if page == "Manufacturing Overview":
        show_manufacturing_overview()
    elif page == "Defect Analysis":
        show_defect_analysis()
    elif page == "Yield Optimization":
        show_yield_optimization()
    elif page == "Quality Control":
        show_quality_control()
    elif page == "MLOps Pipeline":
        from mlops_dashboard import show_mlops_dashboard
        show_mlops_dashboard()

def show_manufacturing_overview():
    """Manufacturing Overview Dashboard Page"""
    st.header("📈 Manufacturing Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Yield",
            value="87.3%",
            delta="2.1%",
            help="Overall good die yield across all wafers"
        )
    
    with col2:
        st.metric(
            label="Total Wafers",
            value="811,457",
            delta="1,234",
            help="Total number of wafers processed"
        )
    
    with col3:
        st.metric(
            label="Defect Rate",
            value="12.7%",
            delta="-1.8%",
            delta_color="inverse",
            help="Percentage of wafers with defects"
        )
    
    with col4:
        st.metric(
            label="Active Lots",
            value="46,293",
            delta="45",
            help="Number of manufacturing lots"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Yield Trend Over Time")
        # Placeholder chart
        dates = pd.date_range('2024-01-01', periods=30, freq='d')
        yield_data = np.random.normal(87, 3, 30)
        
        fig = px.line(
            x=dates, 
            y=yield_data,
            title="Daily Yield Performance",
            labels={'x': 'Date', 'y': 'Yield (%)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Defect Distribution")
        # Placeholder pie chart
        defect_types = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]
        defect_counts = np.random.randint(1000, 10000, len(defect_types))
        
        fig = px.pie(
            values=defect_counts,
            names=defect_types,
            title="Defect Pattern Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_defect_analysis():
    """Defect Analysis Dashboard Page"""
    st.header("🔍 Defect Pattern Analysis")
    
    # Advanced defect analysis with interactive controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("� Defect Frequency by Pattern Type")
        
        # Generate realistic defect data
        defect_data = pd.DataFrame({
            'Defect_Type': ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"],
            'Count': [15234, 12456, 18923, 14567, 22341, 8932, 4521, 3892],
            'Yield_Impact': [-0.15, -0.20, -0.10, -0.12, -0.08, -0.25, -0.30, -0.40],
            'Cost_Per_Wafer': [45.2, 52.8, 38.1, 41.3, 35.7, 68.9, 78.3, 95.6]
        })
        
        fig = px.bar(
            defect_data, 
            x='Defect_Type', 
            y='Count',
            color='Yield_Impact',
            title="Defect Pattern Frequency & Yield Impact",
            color_continuous_scale='RdYlBu_r',
            hover_data=['Cost_Per_Wafer']
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("🗺️ Wafer Spatial Distribution")
        
        # Create synthetic wafer map heatmap
        np.random.seed(42)
        wafer_size = 25
        x = np.arange(wafer_size)
        y = np.arange(wafer_size)
        X, Y = np.meshgrid(x, y)
        
        # Create realistic defect pattern (edge-heavy)
        center_x, center_y = wafer_size//2, wafer_size//2
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        defect_intensity = np.exp(-distance/8) + 0.3 * np.random.random((wafer_size, wafer_size))
        
        fig = px.imshow(
            defect_intensity,
            title="Defect Intensity Heatmap",
            color_continuous_scale='Reds',
            aspect='equal'
        )
        fig.update_layout(
            xaxis_title="Wafer X Position",
            yaxis_title="Wafer Y Position"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("⚙️ Analysis Controls")
        
        # Interactive controls
        selected_defect = st.selectbox(
            "Focus on Defect Type:",
            defect_data['Defect_Type'].tolist(),
            index=0
        )
        
        time_range = st.selectbox(
            "Time Period:",
            ["Last 7 Days", "Last 30 Days", "Last Quarter", "Last Year"],
            index=1
        )
        
        analysis_type = st.radio(
            "Analysis Mode:",
            ["Frequency", "Severity", "Cost Impact"]
        )
        
        if st.button("🔍 Deep Dive Analysis"):
            st.success(f"Analyzing {selected_defect} patterns for {time_range} using {analysis_type} analysis")
    
    # Advanced analytics section
    st.subheader("🧠 AI-Powered Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Most Critical Defect",
            value="Near-full",
            delta="-$95.6 per wafer",
            delta_color="inverse",
            help="Highest cost impact defect pattern"
        )
        
        st.metric(
            label="Improvement Opportunity",
            value="Random Defects",
            delta="25% reduction possible",
            help="Pattern with highest optimization potential"
        )
    
    with col2:
        # Root cause analysis
        st.markdown("**🔧 Recommended Actions:**")
        st.markdown("""
        1. **Process Temperature**: Reduce by 5°C to minimize center defects
        2. **Chamber Cleaning**: Increase frequency to reduce random patterns  
        3. **Edge Protection**: Implement better wafer handling for edge defects
        4. **Equipment Calibration**: Weekly checks to prevent systematic issues
        """)
    
    # Correlation analysis
    st.subheader("📈 Process Parameter Correlation")
    
    # Generate correlation data
    params = ['Temperature', 'Pressure', 'Flow_Rate', 'Time', 'Humidity']
    defects = ['Center', 'Edge-Loc', 'Random', 'Scratch']
    
    correlation_matrix = np.random.uniform(-0.8, 0.8, (len(params), len(defects)))
    
    fig = px.imshow(
        correlation_matrix,
        x=defects,
        y=params,
        title="Parameter-Defect Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_yield_optimization():
    """Yield Optimization Dashboard Page"""
    st.header("⚡ Yield Optimization")
    
    # Key optimization metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Potential Yield Gain",
            value="+4.2%",
            delta="$2.3M annually",
            help="Estimated yield improvement through optimization"
        )
    
    with col2:
        st.metric(
            label="Optimization Score",
            value="78/100",
            delta="+12 points",
            help="Current process optimization level"
        )
    
    with col3:
        st.metric(
            label="Cost Savings",
            value="$156K/month",
            delta="+18%",
            help="Monthly savings from process improvements"
        )
    
    with col4:
        st.metric(
            label="Defect Reduction",
            value="-23%",
            delta="vs. baseline",
            delta_color="inverse",
            help="Reduction in defect rates"
        )
    
    # Main optimization dashboard
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🎯 Parameter Sensitivity Analysis")
        
        # Generate sensitivity data
        parameters = ['Temperature', 'Pressure', 'Etch_Time', 'Flow_Rate', 'Chamber_Pressure']
        sensitivity = np.random.uniform(0.1, 0.9, len(parameters))
        current_values = [1050, 10.2, 118, 195, 1.45]
        optimal_values = [1045, 9.8, 125, 205, 1.52]
        
        # Display sensitivity metrics
        st.write("**Parameter Sensitivity Scores:**")
        for i, param in enumerate(parameters):
            st.write(f"• {param}: {sensitivity[i]:.2f}")
        
        fig = go.Figure()
        
        # Add sensitivity bars
        fig.add_trace(go.Bar(
            x=parameters,
            y=sensitivity,
            name='Sensitivity Score',
            marker_color='lightblue',
            yaxis='y2'
        ))
        
        # Current vs optimal parameters
        fig.add_trace(go.Scatter(
            x=parameters,
            y=current_values,
            mode='markers+lines',
            name='Current Settings',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=parameters,
            y=optimal_values,
            mode='markers+lines',
            name='Optimized Settings',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Process Parameter Optimization",
            xaxis_title="Parameters",
            yaxis_title="Values (Normalized)",
            yaxis2=dict(
                title="Sensitivity Score",
                overlaying='y',
                side='right'
            ),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization recommendations
        st.subheader("🚀 Optimization Recommendations")
        
        recommendations = [
            {"Parameter": "Temperature", "Current": "1050°C", "Optimal": "1045°C", "Impact": "+1.2% yield", "Priority": "High"},
            {"Parameter": "Flow Rate", "Current": "195 sccm", "Optimal": "205 sccm", "Impact": "+0.8% yield", "Priority": "Medium"},
            {"Parameter": "Etch Time", "Current": "118 sec", "Optimal": "125 sec", "Impact": "+0.6% yield", "Priority": "Medium"},
            {"Parameter": "Pressure", "Current": "10.2 Torr", "Optimal": "9.8 Torr", "Impact": "+1.1% yield", "Priority": "High"},
        ]
        
        df_recommendations = pd.DataFrame(recommendations)
        
        # Color code by priority
        def color_priority(val):
            if val == 'High':
                return 'background-color: #ffebee'
            elif val == 'Medium':
                return 'background-color: #fff3e0'
            return ''
        
        # Apply styling to the dataframe
        styled_df = df_recommendations.style.map(color_priority, subset=['Priority'])
        
        st.dataframe(
            styled_df,
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Yield Prediction Model")
        
        # ML model performance metrics
        st.metric("Model Accuracy", "94.2%", "+2.1%")
        st.metric("Prediction Confidence", "89.7%", "+1.8%")
        
        # Feature importance
        features = ['Temperature', 'Pressure', 'Time', 'Flow', 'Humidity']
        importance = [0.28, 0.24, 0.19, 0.16, 0.13]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Feature Importance in Yield Prediction",
            color=importance,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization controls
        st.subheader("🎛️ Optimization Controls")
        
        optimization_mode = st.selectbox(
            "Optimization Target:",
            ["Maximize Yield", "Minimize Cost", "Balanced Approach"]
        )
        
        risk_tolerance = st.slider(
            "Risk Tolerance:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="Higher values allow more aggressive optimization"
        )
        
        if st.button("🚀 Run Optimization"):
            with st.spinner("Running optimization algorithm..."):
                import time
                time.sleep(2)
                st.success(f"Optimization complete! Mode: {optimization_mode}, Risk: {risk_tolerance:.1f}")
                st.info("💡 Recommended: Adjust temperature to 1045°C for +2.3% yield improvement")
                time.sleep(2)  # Simulate processing
                st.success("Optimization complete! New parameters calculated.")
                st.balloons()
    
    # ROI Analysis
    st.subheader("💰 Return on Investment Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Yield Improvements**")
        yield_improvements = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Baseline_Yield': [85.2, 85.1, 84.9, 85.3, 85.0, 85.1],
            'Optimized_Yield': [87.1, 87.8, 88.2, 88.9, 89.1, 89.4]
        })
        
        fig = px.line(
            yield_improvements,
            x='Month',
            y=['Baseline_Yield', 'Optimized_Yield'],
            title="Yield Improvement Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**💵 Cost Impact**")
        cost_data = {
            'Category': ['Material Savings', 'Reduced Rework', 'Increased Throughput', 'Energy Savings'],
            'Monthly_Savings': [45000, 78000, 123000, 23000]
        }
        
        fig = px.pie(
            values=cost_data['Monthly_Savings'],
            names=cost_data['Category'],
            title="Monthly Cost Savings Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("**⏱️ Implementation Timeline**")
        timeline_data = pd.DataFrame({
            'Phase': ['Analysis', 'Testing', 'Implementation', 'Validation'],
            'Duration_Weeks': [2, 4, 6, 3],
            'Status': ['Complete', 'In Progress', 'Planned', 'Planned']
        })
        
        fig = px.bar(
            timeline_data,
            x='Phase',
            y='Duration_Weeks',
            color='Status',
            title="Optimization Implementation Timeline"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_quality_control():
    """Quality Control Dashboard Page"""
    st.header("🎯 Quality Control")
    
    # SPC Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Process Capability",
            value="Cpk = 1.67",
            delta="+0.12",
            help="Process capability index (target > 1.33)"
        )
    
    with col2:
        st.metric(
            label="Control Status",
            value="In Control",
            delta="✅ Stable",
            help="Statistical process control status"
        )
    
    with col3:
        st.metric(
            label="Sigma Level",
            value="4.2σ",
            delta="+0.3σ",
            help="Process sigma level (higher is better)"
        )
    
    with col4:
        st.metric(
            label="Defects Per Million",
            value="127 DPM",
            delta="-23 DPM",
            delta_color="inverse",
            help="Defects per million opportunities"
        )
    
    # Control Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistical Process Control Charts")
        
        # Generate SPC data
        np.random.seed(42)
        n_points = 50
        process_mean = 87.5
        process_std = 2.1
        
        # X-bar chart data
        sample_means = np.random.normal(process_mean, process_std/np.sqrt(5), n_points)
        sample_numbers = list(range(1, n_points + 1))
        
        # Control limits
        ucl = process_mean + 3 * (process_std/np.sqrt(5))
        lcl = process_mean - 3 * (process_std/np.sqrt(5))
        
        fig = go.Figure()
        
        # Sample points
        fig.add_trace(go.Scatter(
            x=sample_numbers,
            y=sample_means,
            mode='markers+lines',
            name='Sample Means',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Control limits
        fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL")
        fig.add_hline(y=process_mean, line_dash="solid", line_color="green", annotation_text="Target")
        fig.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL")
        
        # Mark out-of-control points
        ooc_points = sample_means > ucl
        if any(ooc_points):
            fig.add_trace(go.Scatter(
                x=[i+1 for i, val in enumerate(ooc_points) if val],
                y=[val for val in sample_means if val > ucl],
                mode='markers',
                name='Out of Control',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title="X-bar Control Chart - Yield Performance",
            xaxis_title="Sample Number",
            yaxis_title="Yield (%)",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Anomaly Detection")
        
        # Anomaly detection visualization
        time_points = pd.date_range('2024-01-01', periods=100, freq='h')
        normal_data = np.random.normal(85, 2, 90)
        anomaly_data = np.random.normal(78, 3, 10)  # Anomalous points
        
        # Combine data
        all_data = np.concatenate([normal_data, anomaly_data])
        np.random.shuffle(all_data)
        
        # Identify anomalies (simple threshold-based)
        threshold = np.mean(all_data) - 2 * np.std(all_data)
        is_anomaly = all_data < threshold
        
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=time_points[~is_anomaly],
            y=all_data[~is_anomaly],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4)
        ))
        
        # Anomalous points
        if any(is_anomaly):
            fig.add_trace(go.Scatter(
                x=time_points[is_anomaly],
                y=all_data[is_anomaly],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
        
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text="Anomaly Threshold")
        
        fig.update_layout(
            title="Real-time Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Process Value",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Process monitoring dashboard
    st.subheader("🔍 Real-time Process Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌡️ Temperature Monitoring**")
        
        temp_data = pd.DataFrame({
            'Equipment': ['Chamber_A', 'Chamber_B', 'Chamber_C', 'Chamber_D'],
            'Current_Temp': [1048, 1052, 1049, 1051],
            'Target_Temp': [1050, 1050, 1050, 1050],
            'Status': ['✅ Normal', '⚠️ High', '✅ Normal', '✅ Normal']
        })
        
        st.dataframe(temp_data, use_container_width=True)
        
        # Temperature trend
        hours = list(range(24))
        temp_trend = 1050 + 2 * np.sin(np.array(hours) * np.pi / 12) + np.random.normal(0, 0.5, 24)
        
        fig = px.line(x=hours, y=temp_trend, title="24-Hour Temperature Trend")
        fig.update_layout(xaxis_title="Hour", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**⚡ Equipment Health**")
        
        equipment_health = pd.DataFrame({
            'Equipment': ['Etcher_1', 'Etcher_2', 'CVD_1', 'CVD_2', 'Lithography'],
            'Health_Score': [95, 87, 92, 89, 98],
            'Uptime': ['99.2%', '96.8%', '98.1%', '97.3%', '99.8%'],
            'Next_Maintenance': ['12 days', '3 days', '8 days', '5 days', '20 days']
        })
        
        st.dataframe(equipment_health, use_container_width=True)
        
        # Health score visualization
        fig = px.bar(
            equipment_health,
            x='Equipment',
            y='Health_Score',
            title="Equipment Health Scores",
            color='Health_Score',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("**📈 Quality Trends**")
        
        # Quality metrics over time
        days = pd.date_range('2024-01-01', periods=30, freq='d')
        yield_trend = 85 + 5 * np.sin(np.arange(30) * 2 * np.pi / 7) + np.random.normal(0, 1, 30)
        defect_trend = 15 - 5 * np.sin(np.arange(30) * 2 * np.pi / 7) + np.random.normal(0, 0.5, 30)
        
        quality_df = pd.DataFrame({
            'Date': days,
            'Yield': yield_trend,
            'Defect_Rate': defect_trend
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=quality_df['Date'], y=quality_df['Yield'], name="Yield %"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=quality_df['Date'], y=quality_df['Defect_Rate'], name="Defect Rate %"),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Yield (%)", secondary_y=False)
        fig.update_yaxes(title_text="Defect Rate (%)", secondary_y=True)
        
        fig.update_layout(title="Quality Metrics Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    # Alert system
    st.subheader("🚨 Alert Management System")
    
    alerts_data = [
        {"Time": "2024-07-27 06:45", "Equipment": "Chamber_B", "Alert": "Temperature High", "Severity": "Medium", "Status": "Active"},
        {"Time": "2024-07-27 05:32", "Equipment": "Etcher_2", "Alert": "Pressure Deviation", "Severity": "Low", "Status": "Resolved"},
        {"Time": "2024-07-27 04:18", "Equipment": "CVD_1", "Alert": "Gas Flow Warning", "Severity": "High", "Status": "Active"},
        {"Time": "2024-07-27 03:55", "Equipment": "Lithography", "Alert": "Alignment Check", "Severity": "Low", "Status": "Acknowledged"},
    ]
    
    alerts_df = pd.DataFrame(alerts_data)
    
    # Color-code by severity
    def color_severity(row):
        if row['Severity'] == 'High':
            return ['background-color: #ffebee'] * len(row)
        elif row['Severity'] == 'Medium':
            return ['background-color: #fff3e0'] * len(row)
        else:
            return ['background-color: #e8f5e8'] * len(row)
    
    # Apply styling and display
    styled_alerts = alerts_df.style.apply(color_severity, axis=1)
    st.dataframe(styled_alerts, use_container_width=True)
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Schedule Maintenance"):
            st.success("Maintenance scheduled for Chamber_B")
    
    with col2:
        if st.button("📧 Send Alert Report"):
            st.success("Alert report sent to engineering team")
    
    with col3:
        if st.button("📊 Generate QC Report"):
            st.success("Quality control report generated")

if __name__ == "__main__":
    main()
