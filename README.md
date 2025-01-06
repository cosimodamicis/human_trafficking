# Human Trafficking Analysis Dashboard

## Overview
This interactive dashboard analyzes patterns in human trafficking using the CTDC synthetic dataset. Built with Streamlit, it provides comprehensive visualizations and statistical analysis tools to understand trafficking patterns while protecting victim privacy through the use of synthetic data.

## Features

### 📊 Interactive Filtering
- Time period selection
- Geographic filters (Regions of Origin and Exploitation)
- Demographic filters (Gender)
- Exploitation type filters
- Advanced data completeness filters

### 📈 Visualizations
- Victim-Perpetrator Relationship Analysis
  - Distribution of relationships
  - Exploitation types by relationship

- Geographic Analysis
  - Interactive Sankey diagram showing trafficking flows
  - Region-wise distribution analysis

- Perpetrator Analysis
  - Distribution of perpetrator roles
  - Gender distribution of perpetrators

- Temporal Analysis
  - Time-series visualization of exploitation types
  - Trend analysis over years

### 🔍 Statistical Analysis
- Chi-square analysis of key relationships
- Pattern strength visualization through heatmaps
- Detailed breakdown of significant patterns
- User-friendly interpretation of statistical findings

### 💡 Data Quality Features
- Missing value analysis
- Data completeness filters
- Unknown value handling options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/human-trafficking-analysis.git
cd human-trafficking-analysis
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your CTDC synthetic dataset file (`CTDC_VPsynthetic_condensed.csv`) in the project directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the dashboard in your web browser at `http://localhost:8501`

## Required Dependencies
- streamlit
- pandas
- numpy
- plotly
- scipy
- python-dotenv (if using environment variables)

## Data Description
This dashboard uses synthetic data generated from the Counter Trafficking Data Collaborative (CTDC) dataset. The synthetic data maintains statistical patterns while ensuring complete privacy protection. Key variables include:

- Temporal data (year of registration)
- Geographic information (regions of origin and exploitation)
- Demographic data (gender)
- Exploitation types (forced labor, sexual exploitation)
- Perpetrator information
- Victim-perpetrator relationships

## Features Guide

### Filtering Options
- Use sidebar filters to focus on specific subsets of data
- Toggle "Treat 'Unknown' values as missing data" for different handling of unknown values
- Select data completeness requirements by variable group

### Visualization Sections
1. **Data Quality Overview**
   - View missing value distributions
   - Track impact of applied filters

2. **Victim-Perpetrator Analysis**
   - Explore relationship patterns
   - Analyze exploitation types by relationship

3. **Geographical Analysis**
   - Visualize trafficking flows between regions
   - Identify major origin-destination patterns

4. **Pattern Analysis**
   - Statistical significance of relationships
   - Visual representation of pattern strength
   - Detailed breakdown of significant findings

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT License](LICENSE)

## Acknowledgments
- Counter Trafficking Data Collaborative (CTDC) for the synthetic dataset
- Streamlit for the web application framework
- Plotly for interactive visualizations

## Contact
For questions or support, please [open an issue](https://github.com/yourusername/human-trafficking-analysis/issues) on this repository.