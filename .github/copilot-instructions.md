<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for Global Foundries Wafer Manufacturing Project

## Project Context

This project focuses on semiconductor wafer manufacturing optimization using the WM-811K dataset for Global Foundries. The goal is to create an impressive dashboard that demonstrates expertise in:

- Semiconductor manufacturing processes
- Defect pattern analysis and classification
- Yield optimization and process improvement
- Industrial data science and visualization

## Code Generation Guidelines

### Data Science Best Practices

- Always include proper error handling for data loading and processing
- Use efficient pandas operations and vectorization when possible
- Implement proper data validation and quality checks
- Follow the 80/20 rule for train/test splits in ML models
- Use cross-validation for model evaluation

### Semiconductor Domain Knowledge

- Understand that wafer maps represent spatial defect patterns on silicon wafers
- Defect patterns include: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full
- Yield is calculated as (good dies / total dies) on each wafer
- Manufacturing lots represent batches of wafers processed together
- Process parameters affect defect formation and should be correlated with yield

### Dashboard Development

- Use Streamlit for interactive dashboards with caching decorators
- Implement responsive layouts that work on different screen sizes
- Create modular components for reusability
- Include loading indicators for long-running computations
- Provide clear explanations and tooltips for semiconductor terminology

### Visualization Standards

- Use consistent color schemes appropriate for manufacturing data
- Include proper legends, axis labels, and titles
- Implement interactive features like filtering and drill-down capabilities
- Show confidence intervals and uncertainty where appropriate
- Use heatmaps for spatial defect distribution analysis

### Machine Learning Approach

- Start with baseline models (Random Forest, SVM) before deep learning
- Use appropriate metrics for imbalanced classification (F1-score, AUC-ROC)
- Implement feature engineering specific to wafer map data
- Consider ensemble methods for improved robustness
- Document model assumptions and limitations

### Performance Optimization

- Use caching for expensive computations (@st.cache_data for Streamlit)
- Implement efficient image processing for wafer map visualizations
- Use batch processing for large datasets
- Optimize memory usage when dealing with 811K images

### Industry Relevance

- Focus on actionable insights that manufacturing engineers can use
- Provide cost-benefit analysis for process improvements
- Include statistical significance testing for process changes
- Consider real-world constraints like equipment limitations
- Emphasize ROI and business impact in recommendations

## File Organization

- Keep data processing functions in `src/data_processing.py`
- ML models and evaluation in `src/defect_classifier.py`
- Yield analysis functions in `src/yield_analyzer.py`
- Dashboard components in `dashboard/` directory
- Jupyter notebooks for exploration and prototyping

## Code Quality

- Use type hints for function parameters and returns
- Include comprehensive docstrings with semiconductor context
- Follow PEP 8 style guidelines
- Add unit tests for critical functions
- Use logging for debugging and monitoring
