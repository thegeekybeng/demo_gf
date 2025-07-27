# Global Foundries Wafer Manufacturing Optimization Dashboard

## 🎯 Project Overview

A comprehensive manufacturing process optimization dashboard analyzing the WM-811K wafer map dataset to improve semiconductor fabrication yield and identify process optimization opportunities for Global Foundries.

## 🔬 Dataset

- **WM-811K Wafer Map Dataset**: 811,457 wafer map images from real semiconductor manufacturing
- **8 Defect Pattern Classes**: Various failure patterns in semiconductor wafers
- **46,293 Manufacturing Lots**: Real production data for comprehensive analysis

## 🚀 Key Features

### 1. Yield Analysis Dashboard

- Real-time yield tracking across different lots and time periods
- Statistical process control (SPC) charts
- Yield trend analysis and forecasting

### 2. Defect Pattern Intelligence

- Automated defect classification using computer vision
- Pattern frequency analysis and correlation studies
- Root cause analysis for common defect patterns

### 3. Process Optimization Tools

- Parameter correlation analysis
- Predictive models for yield optimization
- Recommendations for process parameter adjustments

### 4. Interactive Visualizations

- Heat maps of defect distributions across wafer locations
- Time-series analysis of manufacturing performance
- Comparative analysis between different production runs

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original WM-811K dataset
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_defect_analysis.ipynb
│   ├── 03_yield_optimization.ipynb
│   └── 04_model_development.ipynb
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── defect_classifier.py    # ML models for defect classification
│   ├── yield_analyzer.py       # Yield analysis functions
│   └── visualization.py        # Custom plotting functions
├── dashboard/
│   ├── app.py                  # Main Streamlit dashboard
│   ├── pages/                  # Dashboard pages
│   └── components/             # Reusable dashboard components
└── requirements.txt
```

## 🛠️ Installation & Setup

1. **Clone and setup environment:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download WM-811K dataset:**

   - Download from Kaggle: https://www.kaggle.com/datasets/muhammedjunayed/wm811k-silicon-wafer-map-dataset-image/data
   - Extract to `data/raw/` directory

3. **Run the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## 🌐 Live Demo

**📱 Access the live dashboard**: [Deployment URL will be here after Streamlit Cloud deployment]

**🔗 GitHub Repository**: https://github.com/thegeekybeng/demo_gf

## 📊 Dashboard Sections

### Manufacturing Overview

- Overall yield metrics and KPIs
- Production volume and throughput analysis
- Real-time process monitoring

### Defect Analysis

- Defect pattern classification and frequency
- Spatial distribution of defects across wafers
- Correlation between defect patterns and process parameters

### Yield Optimization

- Predictive models for yield forecasting
- Process parameter sensitivity analysis
- Recommendations for process improvements

### Quality Control

- Statistical process control charts
- Anomaly detection in manufacturing processes
- Automated alerts for process deviations

## 🎯 Business Impact for Global Foundries

1. **Increased Yield**: Identify and eliminate sources of defects
2. **Reduced Costs**: Optimize process parameters to minimize waste
3. **Faster Troubleshooting**: Automated defect pattern recognition
4. **Data-Driven Decisions**: Evidence-based process optimization
5. **Predictive Maintenance**: Early detection of equipment issues

## 🔧 Technologies Used

- **Python**: Core programming language
- **Streamlit**: Interactive dashboard framework
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn/XGBoost**: Machine learning models
- **Plotly/Seaborn**: Advanced visualizations
- **OpenCV**: Image processing for wafer maps

## 📈 Future Enhancements

- Real-time data integration with fab equipment
- Advanced deep learning models for defect detection
- Federated learning across multiple manufacturing sites
- Integration with existing MES/ERP systems

---

**Developed for Global Foundries Recruitment Showcase**
_Demonstrating expertise in semiconductor manufacturing, data science, and industrial optimization_
