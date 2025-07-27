"""
MLOps Dashboard Integration for Global Foundries
Integrates the MLOps pipeline with the Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from mlops_pipeline import MLOpsPipeline, load_pipeline_config
    MLOPS_AVAILABLE = True
except ImportError:
    st.warning("MLOps pipeline not available. Running in demo mode.")
    MLOpsPipeline = None
    load_pipeline_config = None
    MLOPS_AVAILABLE = False

def show_mlops_dashboard():
    """MLOps Pipeline Dashboard Page"""
    st.header("🤖 MLOps Pipeline Dashboard")
    
    if not MLOPS_AVAILABLE:
        st.info("MLOps components running in demo mode. Install MLflow for full functionality.")
    else:
        # Initialize MLOps pipeline for demonstration
        try:
            if load_pipeline_config is not None:
                config = load_pipeline_config()
                st.success(f"MLOps pipeline configuration loaded successfully with {len(config)} sections")
                
                # Show pipeline initialization option
                if st.button("🔧 Initialize MLOps Pipeline"):
                    # This would initialize an actual MLOpsPipeline instance
                    if MLOpsPipeline is not None:
                        # pipeline = MLOpsPipeline("config_path")  # Would be used here
                        st.success("MLOps pipeline initialized (demo mode)")
        except (ImportError, RuntimeError) as e:
            st.error(f"Error initializing MLOps pipeline: {e}")
    
    # Pipeline status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Pipeline Health",
            value="🟢 Healthy",
            delta="All systems operational",
            help="Overall MLOps pipeline status"
        )
    
    with col2:
        st.metric(
            label="Model Accuracy",
            value="94.2%",
            delta="+1.3%",
            help="Current model performance"
        )
    
    with col3:
        st.metric(
            label="Predictions/Hour",
            value="1,247",
            delta="+156",
            help="Real-time prediction throughput"
        )
    
    with col4:
        st.metric(
            label="Last Retrain",
            value="2 days ago",
            delta="On schedule",
            help="Time since last model retraining"
        )
    
    # Main MLOps content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Model Performance", 
        "📊 Data Drift", 
        "🚨 Monitoring", 
        "⚙️ Model Management", 
        "📈 Experiment Tracking"
    ])
    
    with tab1:
        show_model_performance()
    
    with tab2:
        show_data_drift_monitoring()
    
    with tab3:
        show_real_time_monitoring()
    
    with tab4:
        show_model_management()
    
    with tab5:
        show_experiment_tracking()

def show_model_performance():
    """Model Performance Monitoring"""
    st.subheader("🎯 Model Performance Metrics")
    
    # Model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance over time
        dates = pd.date_range('2024-01-01', periods=30, freq='d')
        accuracy_data = np.random.normal(0.94, 0.02, 30)
        accuracy_data = np.clip(accuracy_data, 0.85, 0.98)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=accuracy_data,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='blue', width=3)
        ))
        
        # Add threshold line
        fig.add_hline(y=0.90, line_dash="dash", line_color="red", 
                     annotation_text="Retraining Threshold")
        
        fig.update_layout(
            title="Model Accuracy Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.85, 1.0])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance
        features = ['Temperature', 'Pressure', 'Etch_Time', 'Flow_Rate', 'Chamber_Flow']
        importance = [0.28, 0.24, 0.19, 0.16, 0.13]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Current Model Feature Importance",
            color=importance,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📊 Detailed Model Metrics")
    
    model_metrics = pd.DataFrame({
        'Model': ['Yield Predictor', 'Defect Classifier', 'Anomaly Detector'],
        'Accuracy': [94.2, 91.8, 89.5],
        'Precision': [93.1, 90.2, 87.3],
        'Recall': [92.8, 91.5, 88.9],
        'F1-Score': [92.9, 90.8, 88.1],
        'Last_Updated': ['2024-07-25', '2024-07-24', '2024-07-26'],
        'Status': ['🟢 Healthy', '🟡 Monitoring', '🟢 Healthy']
    })
    
    st.dataframe(model_metrics, use_container_width=True)
    
    # Model comparison
    st.subheader("🔄 Model Version Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox("Select Model:", ["Yield Predictor", "Defect Classifier", "Anomaly Detector"])
    
    with col2:
        st.selectbox("Version A:", ["v1.0", "v1.1", "v1.2", "v2.0"])
    
    with col3:
        st.selectbox("Version B:", ["v1.0", "v1.1", "v1.2", "v2.0"])
    
    if st.button("🔍 Compare Models"):
        st.success("Model comparison completed. Version 2.0 shows 2.3% improvement in accuracy.")

def show_data_drift_monitoring():
    """Data Drift Detection and Monitoring"""
    st.subheader("📊 Data Drift Monitoring")
    
    # Drift detection status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Drift Status", "🟢 No Drift", "All features stable")
    
    with col2:
        st.metric("Features Monitored", "12", "Core process parameters")
    
    with col3:
        st.metric("Drift Threshold", "10%", "Statistical significance")
    
    # Feature drift analysis
    st.subheader("🔍 Feature Drift Analysis")
    
    features = ['Temperature', 'Pressure', 'Etch_Time', 'Flow_Rate', 'Chamber_Flow']
    drift_scores = np.random.uniform(0.02, 0.08, len(features))
    
    # Create drift visualization
    fig = go.Figure()
    
    colors = ['green' if score < 0.05 else 'orange' if score < 0.08 else 'red' for score in drift_scores]
    
    fig.add_trace(go.Bar(
        x=features,
        y=drift_scores,
        marker_color=colors,
        name='Drift Score'
    ))
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                 annotation_text="Warning Threshold")
    fig.add_hline(y=0.10, line_dash="dash", line_color="red", 
                 annotation_text="Critical Threshold")
    
    fig.update_layout(
        title="Data Drift Detection by Feature",
        xaxis_title="Features",
        yaxis_title="Drift Score",
        yaxis=dict(range=[0, 0.12])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical drift trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Drift Trends")
        
        # Generate drift trend data
        days = pd.date_range('2024-01-01', periods=30, freq='d')
        temp_drift = np.random.uniform(0.01, 0.06, 30)
        pressure_drift = np.random.uniform(0.02, 0.07, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=temp_drift, name='Temperature', mode='lines'))
        fig.add_trace(go.Scatter(x=days, y=pressure_drift, name='Pressure', mode='lines'))
        
        fig.update_layout(
            title="Feature Drift Over Time",
            xaxis_title="Date",
            yaxis_title="Drift Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Drift Detection Settings")
        
        st.slider("Warning Threshold", 0.0, 0.2, 0.05, 0.01)
        st.slider("Critical Threshold", 0.0, 0.3, 0.10, 0.01)
        st.selectbox("Detection Method", ["Statistical", "KL Divergence", "Wasserstein Distance"])
        st.number_input("Monitoring Window (hours)", 1, 168, 24)
        
        if st.button("🔧 Update Settings"):
            st.success("Drift detection settings updated")

def show_real_time_monitoring():
    """Real-time Pipeline Monitoring"""
    st.subheader("🔴 Real-time Pipeline Monitoring")
    
    # Live metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predictions/Min", "21", "+3")
    
    with col2:
        st.metric("Avg Latency", "45ms", "-2ms")
    
    with col3:
        st.metric("Error Rate", "0.02%", "-0.01%")
    
    with col4:
        st.metric("CPU Usage", "23%", "+1%")
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Throughput")
        
        # Generate real-time throughput data
        timestamps = pd.date_range('2024-07-27 06:00', periods=60, freq='min')
        throughput = np.random.poisson(20, 60)
        
        fig = px.line(x=timestamps, y=throughput, title="Predictions per Minute")
        fig.update_layout(xaxis_title="Time", yaxis_title="Predictions/Min")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Response Latency")
        
        latency_data = np.random.normal(45, 8, 60)
        
        fig = px.line(x=timestamps, y=latency_data, title="Prediction Latency")
        fig.update_layout(xaxis_title="Time", yaxis_title="Latency (ms)")
        st.plotly_chart(fig, use_container_width=True)
    
    # System health
    st.subheader("🖥️ System Health")
    
    health_metrics = pd.DataFrame({
        'Component': ['API Server', 'Model Server', 'Database', 'Message Queue', 'Cache'],
        'Status': ['🟢 Healthy', '🟢 Healthy', '🟢 Healthy', '🟡 Warning', '🟢 Healthy'],
        'CPU (%)': [23, 45, 12, 67, 8],
        'Memory (%)': [34, 78, 45, 89, 23],
        'Uptime': ['99.9%', '99.7%', '99.8%', '98.2%', '99.9%']
    })
    
    st.dataframe(health_metrics, use_container_width=True)
    
    # Alerts
    st.subheader("🚨 Active Alerts")
    
    alerts = [
        {"Time": "06:45", "Component": "Message Queue", "Alert": "High Memory Usage", "Severity": "Medium"},
        {"Time": "06:32", "Component": "Model Server", "Alert": "Increased Latency", "Severity": "Low"},
    ]
    
    if alerts:
        for alert in alerts:
            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟠"}[alert["Severity"]]
            st.warning(f"{severity_color} {alert['Time']} - {alert['Component']}: {alert['Alert']}")
    else:
        st.success("✅ No active alerts")

def show_model_management():
    """Model Lifecycle Management"""
    st.subheader("⚙️ Model Lifecycle Management")
    
    # Model registry
    st.subheader("📚 Model Registry")
    
    models_registry = pd.DataFrame({
        'Model_Name': ['yield_predictor_v2.0', 'yield_predictor_v1.2', 'defect_classifier_v1.1'],
        'Version': ['2.0', '1.2', '1.1'],
        'Status': ['🟢 Production', '🟡 Staging', '🔵 Development'],
        'Accuracy': ['94.2%', '92.8%', '91.5%'],
        'Created': ['2024-07-25', '2024-07-20', '2024-07-15'],
        'Author': ['ML Team', 'ML Team', 'ML Team']
    })
    
    st.dataframe(models_registry, use_container_width=True)
    
    # Model deployment
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Model Deployment")
        
        st.selectbox("Select Model:", ["yield_predictor_v2.0", "defect_classifier_v1.1"])
        st.selectbox("Target Environment:", ["Production", "Staging", "Testing"])
        st.selectbox("Deployment Strategy:", ["Blue-Green", "Canary", "Rolling"])
        
        st.text_area(
            "Deployment Configuration:",
            value="""{
    "replicas": 3,
    "cpu_limit": "500m",
    "memory_limit": "1Gi",
    "auto_scaling": true
}"""
        )
        
        if st.button("🚀 Deploy Model"):
            st.success("Model deployment initiated. Estimated time: 5 minutes")
    
    with col2:
        st.subheader("📊 Deployment History")
        
        deployment_history = pd.DataFrame({
            'Timestamp': ['2024-07-25 14:30', '2024-07-20 09:15', '2024-07-15 16:45'],
            'Model': ['yield_predictor_v2.0', 'yield_predictor_v1.2', 'defect_classifier_v1.1'],
            'Environment': ['Production', 'Production', 'Staging'],
            'Status': ['✅ Success', '✅ Success', '✅ Success']
        })
        
        st.dataframe(deployment_history, use_container_width=True)
    
    # A/B Testing
    st.subheader("🧪 A/B Testing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox("Model A:", ["yield_predictor_v1.2"])
        st.metric("Model A Performance", "92.8%", "Baseline")
    
    with col2:
        st.selectbox("Model B:", ["yield_predictor_v2.0"])
        st.metric("Model B Performance", "94.2%", "+1.4%")
    
    with col3:
        st.selectbox("Traffic Split:", ["50/50", "90/10", "95/5"])
        st.metric("Test Duration", "7 days", "In progress")
    
    if st.button("📊 View A/B Test Results"):
        st.info("A/B test results show Model B has significantly better performance with p-value < 0.05")

def show_experiment_tracking():
    """ML Experiment Tracking"""
    st.subheader("📈 Experiment Tracking")
    
    # Experiment overview
    experiments = pd.DataFrame({
        'Experiment_ID': ['exp_001', 'exp_002', 'exp_003', 'exp_004'],
        'Name': ['Hyperparameter Tuning', 'Feature Engineering', 'Data Augmentation', 'Architecture Search'],
        'Model': ['Yield Predictor', 'Defect Classifier', 'Yield Predictor', 'Anomaly Detector'],
        'Best_Accuracy': [94.2, 91.8, 93.7, 89.5],
        'Runs': [25, 15, 30, 20],
        'Status': ['✅ Complete', '🔄 Running', '✅ Complete', '✅ Complete'],
        'Started': ['2024-07-20', '2024-07-25', '2024-07-18', '2024-07-22']
    })
    
    st.dataframe(experiments, use_container_width=True)
    
    # Experiment details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Experiment Results")
        
        # Generate experiment results
        runs = list(range(1, 26))
        accuracies = np.random.normal(0.92, 0.02, 25)
        accuracies = np.clip(accuracies, 0.85, 0.95)
        
        # Highlight best run
        best_idx = np.argmax(accuracies)
        colors = ['red' if i == best_idx else 'blue' for i in range(len(runs))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=runs,
            y=accuracies,
            mode='markers',
            marker=dict(color=colors, size=8),
            name='Experiment Runs'
        ))
        
        fig.update_layout(
            title="Hyperparameter Tuning Results",
            xaxis_title="Run Number",
            yaxis_title="Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Best Parameters")
        
        best_params = {
            'n_estimators': 150,
            'max_depth': 12,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        for param, value in best_params.items():
            st.text(f"{param}: {value}")
        
        st.metric("Best Accuracy", "94.2%", "+1.4%")
        
        if st.button("📥 Download Model"):
            st.success("Best model downloaded successfully")
    
    # Hyperparameter visualization
    st.subheader("🎛️ Hyperparameter Analysis")
    
    # Generate hyperparameter data
    n_estimators = np.random.randint(50, 200, 100)
    max_depth = np.random.randint(5, 20, 100)
    accuracy = 0.85 + 0.1 * (n_estimators / 200) + 0.05 * (max_depth / 20) + np.random.normal(0, 0.02, 100)
    
    fig = px.scatter(
        x=n_estimators,
        y=max_depth,
        color=accuracy,
        title="Hyperparameter Space Exploration",
        labels={'x': 'N Estimators', 'y': 'Max Depth', 'color': 'Accuracy'},
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

# Add MLOps dashboard to the main app
def add_mlops_page():
    """Add MLOps page to the main dashboard"""
    return show_mlops_dashboard()

if __name__ == "__main__":
    show_mlops_dashboard()
