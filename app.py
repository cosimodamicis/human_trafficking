import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import norm

# Custom styling
st.markdown("""
    <style>
    .stApp {background-color: white}
    .main > div {padding: 2rem}
    .block-container {padding-top: 1rem}
    h1, h2, h3 {background-color: #FFB81C; padding: 0.5rem; color: black}
    .stSidebar {background-color: #f5f5f5}
    </style>
    """, unsafe_allow_html=True)

# Color schemes
colors_pie = ['#FFB81C', '#000000', '#E31837', '#4CAF50']
colors_main = {
    'primary': '#FFB81C',
    'secondary': '#000000',
    'negative': '#E31837',
    'positive': '#4CAF50'
}

def calculate_chi_square_analysis(df, var1, var2):
    """
    Performs chi-square analysis including test statistics, 
    standardized residuals and their confidence intervals
    """
    # Create contingency table
    cont_table = pd.crosstab(df[var1], df[var2])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(cont_table)
    
    # Calculate Cramer's V
    n = cont_table.sum().sum()
    min_dim = min(cont_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Calculate standardized residuals and their CIs
    observed = cont_table.values
    residuals = (observed - expected) / np.sqrt(expected)
    
    # 95% CI for standardized residuals
    ci_95 = norm.ppf(0.975)  # 1.96 for 95% CI
    residual_cis = {
        'lower': residuals - ci_95,
        'upper': residuals + ci_95
    }
    
    # Check assumption (80% of cells should have expected freq > 5)
    total_cells = expected.size
    cells_over_5 = np.sum(expected > 5)
    assumption_met = (cells_over_5 / total_cells) >= 0.8
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramer_v': cramer_v,
        'assumption_met': assumption_met,
        'cells_over_5_pct': (cells_over_5 / total_cells) * 100,
        'residuals': residuals,
        'residual_cis': residual_cis,
        'cont_table': cont_table,
        'expected': expected
    }

def interpret_cramer_v(v):
    """Returns interpretation of Cramer's V effect size"""
    if v < 0.1:
        return "negligible association"
    elif v < 0.2:
        return "weak association"
    elif v < 0.3:
        return "moderate association"
    else:
        return "strong association"

def display_refined_chi_square_results(result, var1_name, var2_name):
    """Display a more user-friendly version of the chi-square results"""
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Visual Summary", "📑 Detailed Findings"])
    
    with tab1:
        # Create and display the heatmap
        residuals = result['residuals']
        cont_table = result['cont_table']
        fig = create_association_heatmap(
            residuals,
            cont_table.index,
            cont_table.columns,
            f"Relationship Patterns: {var1_name} vs {var2_name}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a simple interpretation guide
        st.write("**How to read this visualization:**")
        st.write("""
        - 🔵 Blue cells show combinations that occur more often than expected
        - 🔴 Red cells show combinations that occur less often than expected
        - Darker colors indicate stronger patterns
        """)
        
        # Show key findings
        st.write("**Key Patterns:**")
        
        # Get top 3 positive and negative patterns
        significant_patterns = []
        for i in range(residuals.shape[0]):
            for j in range(residuals.shape[1]):
                if abs(residuals[i, j]) > 1.96:  # Statistically significant
                    significant_patterns.append({
                        'row': cont_table.index[i],
                        'col': cont_table.columns[j],
                        'residual': residuals[i, j]
                    })
        
        # Sort by absolute residual value and get top patterns
        top_patterns = sorted(significant_patterns, 
                            key=lambda x: abs(x['residual']), 
                            reverse=True)[:6]
        
        # Display in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Common Combinations:**")
            for pattern in [p for p in top_patterns if p['residual'] > 0][:3]:
                st.write(f"• {pattern['row']} + {pattern['col']}")
        
        with col2:
            st.write("**Least Common Combinations:**")
            for pattern in [p for p in top_patterns if p['residual'] < 0][:3]:
                st.write(f"• {pattern['row']} + {pattern['col']}")
    
    with tab2:
        # Show relationship strength
        st.write(f"**Overall Relationship Strength:** {interpret_cramer_v(result['cramer_v'])}")
        
        # Create a more detailed table of findings
        if abs(result['cramer_v']) >= 0.1:  # Only show if there's at least a weak association
            significant_patterns_df = pd.DataFrame([
                {
                    'Combination': f"{p['row']} + {p['col']}",
                    'Pattern': 'More common than expected' if p['residual'] > 0 
                             else 'Less common than expected',
                    'Strength': abs(p['residual'])
                }
                for p in significant_patterns
            ]).sort_values('Strength', ascending=False)
            
            st.write("**Detailed Patterns:**")
            st.dataframe(significant_patterns_df)


def create_association_heatmap(residuals, row_names, col_names, title):
    """Create a heatmap visualization of standardized residuals"""
    fig = px.imshow(
        residuals,
        x=col_names,
        y=row_names,
        color_continuous_scale='RdBu',  # Red for negative, Blue for positive
        title=title,
        labels={'color': 'Association Strength'}
    )
    
    # Update layout for better readability
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        title_x=0.5,
    )
    
    return fig

def display_chi_square_results(result, var1_name, var2_name):
    """Displays formatted chi-square test results"""
    st.write(f"📊 Statistical Analysis of {var1_name} vs {var2_name}")
    
    # Display assumption check
    st.write("\n**Assumption Check:**")
    if result['assumption_met']:
        st.write("✓ Assumption met: More than 80% of cells have expected frequency > 5")
    else:
        st.write(f"⚠️ Assumption violated: Only {result['cells_over_5_pct']:.1f}% of cells have expected frequency > 5")
    
    # Display test results
    st.write("\n**Test Results:**")
    st.write(f"- Chi-square statistic: {result['chi2']:.2f}")
    st.write(f"- Degrees of freedom: {result['dof']}")
    st.write(f"- p-value: {result['p_value']:.4f}")
    st.write(f"- Cramer's V: {result['cramer_v']:.2f} ({interpret_cramer_v(result['cramer_v'])})")
    
    # Display significant patterns
    st.write("\n**Significant Patterns (Standardized Residuals with 95% CIs):**")
    residuals = result['residuals']
    cis = result['residual_cis']
    cont_table = result['cont_table']
    
    for i in range(residuals.shape[0]):
        for j in range(residuals.shape[1]):
            if abs(residuals[i, j]) > 1.96:
                row_name = cont_table.index[i]
                col_name = cont_table.columns[j]
                st.write(f"• {row_name} & {col_name}: {residuals[i, j]:.2f} "
                        f"[{cis['lower'][i, j]:.2f}, {cis['upper'][i, j]:.2f}]")
                if residuals[i, j] > 0:
                    st.write("  More cases than expected (significantly positive)")
                else:
                    st.write("  Fewer cases than expected (significantly negative)")
    
    # Add interpretation guide
    st.write("\n📝 **How to interpret:**")
    st.write("• p-value < 0.05 indicates a significant relationship")
    st.write("• Standardized residuals > |1.96| indicate significant patterns")
    st.write("• Larger absolute residuals suggest stronger deviations from expected frequencies")

# Set page config
st.set_page_config(
    page_title="Human Trafficking Analysis Dashboard",
    page_icon="🌍",
    layout="wide"
)

def load_data():
    """
    Load and preprocess the CTDC synthetic dataset
    """
    # Read the CSV file
    df = pd.read_csv('CTDC_VPsynthetic_condensed.csv')
    
    # Create a copy of the original data before filling NaN values
    df_original = df.copy()
    
    # Replace empty strings and NaN with explicit 'Unknown' for yearRegister
    df['yearRegister'] = df['yearRegister'].fillna('Unknown')
    df.loc[df['yearRegister'] == '', 'yearRegister'] = 'Unknown'
    
    # Create exploitation status before handling unknowns
    df['exploitation_status'] = df.apply(
        lambda row: ', '.join([
            'Forced Labour' if row['isForcedLabour'] == 1 else '',
            'Sexual Exploitation' if row['isSexualExploit'] == 1 else ''
        ]).strip(', ') or 'Unknown',
        axis=1
    )
    
    # Store both versions: one with Unknown as valid, one with Unknown as missing
    df_unknown_valid = df.copy()
    df_unknown_missing = df.copy()
    
    # Define all categorical columns including yearRegister and exploitation_status
    categorical_cols = [
        'yearRegister', 'gender', 'UN_COO_Region', 'UN_COE_Region', 
        'IP_ageBroad', 'IP_Gender', 'IP_Relation', 'IP_citizen_UNRegion',
        'exploitation_status'
    ]
    
    # Version with Unknown as valid - replace NaN with "Unknown" for categorical variables
    for col in categorical_cols:
        df_unknown_valid[col] = df_unknown_valid[col].fillna('Unknown')
        df_unknown_valid.loc[df_unknown_valid[col] == '', col] = 'Unknown'
    
    # Version with Unknown as missing - replace "Unknown" with NaN
    for col in categorical_cols:
        df_unknown_missing.loc[df_unknown_missing[col] == 'Unknown', col] = np.nan
        df_unknown_missing.loc[df_unknown_missing[col] == '', col] = np.nan
    
    # Common handling for both versions - binary columns
    binary_cols = ['isForcedLabour', 'isSexualExploit', 'IP_Exploiter', 
                  'IP_RecruiterBroker', 'IP_TransactionProcess', 'IP_ControlAbuseKidnap']
    for df_version in [df_unknown_valid, df_unknown_missing]:
        for col in binary_cols:
            df_version[col] = df_version[col].fillna(0).astype(int)
    
    # Calculate missing value percentages from original data
    missing_stats = df_original.isnull().mean() * 100
    
    # Define column groups for filtering
    column_groups = {
        'Temporal': ['yearRegister'],
        'Demographic': ['gender', 'majorityStatus', 'IP_Gender', 'IP_ageBroad'],
        'Geographic': ['UN_COO_Region', 'UN_COE_Region', 'IP_citizen_UNRegion'],
        'Exploitation': ['exploitation_status', 'isForcedLabour', 'isSexualExploit'],
        'Perpetrator Roles': ['IP_Exploiter', 'IP_RecruiterBroker', 
                            'IP_TransactionProcess', 'IP_ControlAbuseKidnap'],
        'Relationship': ['IP_Relation', 'IP_PayMoney']
    }
    
    return df_unknown_valid, df_unknown_missing, missing_stats, column_groups

def create_missing_values_heatmap(missing_stats):
    """Create a heatmap visualization of missing values"""
    fig = px.imshow(
        missing_stats.values.reshape(1, -1),
        labels=dict(x="Variables", y="", color="Missing %"),
        x=missing_stats.index,
        aspect="auto",
        color_continuous_scale="RdYlBu_r"
    )
    fig.update_layout(
        title="Missing Values Heatmap",
        xaxis_tickangle=-45,
        height=200
    )
    return fig

def filter_by_completeness(df, selected_groups, column_groups):
    """Filter dataframe based on completeness criteria - only complete cases for selected groups"""
    if not selected_groups:
        return df
    
    selected_columns = []
    for group in selected_groups:
        selected_columns.extend(column_groups[group])
    
    # Return only complete cases for selected columns
    return df.dropna(subset=selected_columns)

def create_sankey_diagram(df, source_col, target_col):
    """Create a Sankey diagram showing flow between two columns"""
    # Create source-target pairs
    source_ids = {s: i for i, s in enumerate(df[source_col].dropna().unique())}
    target_ids = {t: i + len(source_ids) for i, t in enumerate(df[target_col].dropna().unique())}
    
    # Create flow counts
    flows = df.dropna(subset=[source_col, target_col]).groupby([source_col, target_col]).size().reset_index(name='count')
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(source_ids.keys()) + list(target_ids.keys()),
        ),
        link=dict(
            source=[source_ids[s] for s in flows[source_col]],
            target=[target_ids[t] for t in flows[target_col]],
            value=flows['count']
        )
    )])
    
    fig.update_layout(title=f"Trafficking Flows: {source_col} to {target_col}")
    return fig

def safe_sort_unique(df, column):
    """
    Safely sort unique values handling both strings and NaN
    Returns only non-NaN values
    """
    # Get unique values
    unique_vals = df[column].dropna().unique()
    # Sort the values
    sorted_vals = sorted(unique_vals)
    return sorted_vals



def main():
    st.title("🌍 Human Trafficking Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes patterns in human trafficking using the CTDC synthetic dataset. 
                
    Data is downloaded from https://www.ctdatacollaborative.org/dataset/global-victim-perpetrator-synthetic-dataset-and-resources/resource/07e89e2d-6e9a-4c88-a29c
                                
    The data is synthetic and generated using differential privacy techniques to protect victim privacy while maintaining statistical patterns.
    """)
    
    # Load data with missing value statistics
    df_unknown_valid, df_unknown_missing, missing_stats, column_groups = load_data()
    
    st.sidebar.header("Filters")
    
    # Add checkbox for Unknown handling
    treat_unknown_as_missing = st.sidebar.checkbox(
        "Treat 'Unknown' values as missing data",
        value=False,
        help="If checked, 'Unknown' values will be treated as missing data in completeness calculations",
        key="unknown_handling"
    )
    
    # Use appropriate dataset based on Unknown handling preference
    working_df = df_unknown_missing if treat_unknown_as_missing else df_unknown_valid
    
    # Create filter sections with expanders
    with st.sidebar.expander("Time Periods", expanded=True):
        year_periods = safe_sort_unique(working_df, 'yearRegister')
        selected_periods = []
        for period in year_periods:
            if st.checkbox(str(period), value=True, key=f"period_{period}"):
                selected_periods.append(period)
    
    with st.sidebar.expander("Regions of Origin", expanded=True):
        regions = safe_sort_unique(working_df, 'UN_COO_Region')
        selected_regions = []
        for region in regions:
            if st.checkbox(region, value=True, key=f"origin_{region}"):
                selected_regions.append(region)
    
    with st.sidebar.expander("Regions of Exploitation", expanded=True):
        regions_exploitation = safe_sort_unique(working_df, 'UN_COE_Region')
        selected_regions_exploit = []
        for region in regions_exploitation:
            if st.checkbox(region, value=True, key=f"exploit_{region}"):
                selected_regions_exploit.append(region)
    
    with st.sidebar.expander("Victim Gender", expanded=True):
        genders = safe_sort_unique(working_df, 'gender')
        selected_genders = []
        for gender in genders:
            if st.checkbox(gender, value=True, key=f"gender_{gender}"):
                selected_genders.append(gender)

    with st.sidebar.expander("Exploitation Type", expanded=True):
        exploitation_types = safe_sort_unique(working_df, 'exploitation_status')
        selected_exploitation_types = []
        for exp_type in exploitation_types:
            if st.checkbox(exp_type, value=True, key=f"exp_{exp_type}"):
                selected_exploitation_types.append(exp_type)
    
    # Data Completeness Section
    st.sidebar.header("Data Completeness Filters")
    st.sidebar.markdown("Select groups where you want complete data only:")
    
    selected_groups = []
    for group in column_groups.keys():
        if st.sidebar.checkbox(group, key=f"complete_{group}"):
            selected_groups.append(group)
    
    # Filter data based on all criteria
    filtered_df = working_df[
        (working_df['yearRegister'].isin(selected_periods)) &
        (working_df['UN_COO_Region'].isin(selected_regions)) &
        (working_df['UN_COE_Region'].isin(selected_regions_exploit)) &
        (working_df['gender'].isin(selected_genders)) &
        (working_df['exploitation_status'].isin(selected_exploitation_types))
    ]
    
    # Apply completeness filtering if any groups are selected
    filtered_df = filter_by_completeness(filtered_df, selected_groups, column_groups)


    # Data Quality Section
    if st.checkbox("Show Data Quality Analysis", value=True, key="show_quality"):
        st.header("📊 Data Quality Overview")
        
        # Missing Values Heatmap
        st.subheader("Missing Values Distribution")
        fig = create_missing_values_heatmap(missing_stats)
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact of filtering
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(working_df):,}")
        with col2:
            st.metric("Filtered Records", f"{len(filtered_df):,}")
        with col3:
            st.metric("Records Retained", 
                     f"{(len(filtered_df)/len(working_df)*100):.1f}%")
    
    # Victim-Perpetrator Relationship Analysis
    st.header("🔄 Victim-Perpetrator Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Relationship distribution
        relation_counts = filtered_df['IP_Relation'].value_counts()
        if not relation_counts.empty:
            fig = px.pie(
                values=relation_counts.values,
                names=relation_counts.index,
                title="Distribution of Victim-Perpetrator Relationships"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No relationship data available for the current selection.")
    
    with col2:
        # Exploitation type by relationship
        if 'isForcedLabour' in filtered_df.columns and 'isSexualExploit' in filtered_df.columns:
            exploitation_by_relation = filtered_df.groupby('IP_Relation').agg({
                'isForcedLabour': 'sum',
                'isSexualExploit': 'sum'
            }).reset_index()
            
            if not exploitation_by_relation.empty:
                fig = go.Figure(data=[
                    go.Bar(name='Forced Labour', 
                          x=exploitation_by_relation['IP_Relation'],
                          y=exploitation_by_relation['isForcedLabour']),
                    go.Bar(name='Sexual Exploitation', 
                          x=exploitation_by_relation['IP_Relation'],
                          y=exploitation_by_relation['isSexualExploit'])
                ])
                fig.update_layout(
                    title="Exploitation Type by Relationship",
                    barmode='group',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No exploitation data available for the current selection.")
        else:
            st.warning("Exploitation type columns not found in the dataset.")
    
    # Geographical Analysis
    st.header("🌐 Geographical Analysis")
    if not filtered_df.empty:
        fig_sankey = create_sankey_diagram(filtered_df, 'UN_COO_Region', 'UN_COE_Region')
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.warning("No geographical data available for the current selection.")
    
    # Perpetrator Analysis
    st.header("👥 Perpetrator Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        # Perpetrator roles
        role_cols = ['IP_RecruiterBroker', 'IP_TransactionProcess', 
                    'IP_ControlAbuseKidnap', 'IP_Exploiter']
        if all(col in filtered_df.columns for col in role_cols):
            role_sums = filtered_df[role_cols].sum()
            
            fig = px.bar(
                x=role_sums.index,
                y=role_sums.values,
                title="Distribution of Perpetrator Roles"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Perpetrator role columns not found in the dataset.")
    
    with col4:
        # Perpetrator demographics
        if 'IP_Gender' in filtered_df.columns:
            perp_gender = filtered_df['IP_Gender'].value_counts()
            if not perp_gender.empty:
                fig = px.pie(
                    values=perp_gender.values,
                    names=perp_gender.index,
                    title="Gender Distribution of Perpetrators"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No gender data available for the current selection.")
        else:
            st.warning("Gender column not found in the dataset.")
    
    # Temporal Analysis
    st.header("📈 Temporal Analysis")
    if not filtered_df.empty:
        temporal_data = filtered_df.groupby('yearRegister').agg({
            'isForcedLabour': 'sum',
            'isSexualExploit': 'sum'
        }).reset_index()
        
        fig = px.bar(
            temporal_data,
            x='yearRegister',
            y=['isForcedLabour', 'isSexualExploit'],
            title="Exploitation Types Over Time",
            barmode='group',
            labels={
                'yearRegister': 'Time Period',
                'value': 'Number of Cases',
                'variable': 'Exploitation Type'
            }
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No temporal data available for the current selection.")
    
    # Optional: Show filtered data
    if st.checkbox("Show Filtered Data", key="show_data"):
        st.dataframe(filtered_df)

    # Statistical Analysis Section
    if st.checkbox("Show Statistical Analysis", key="show_stats"):
        st.header("🔍 Pattern Analysis")
        
        st.write("""
        This section reveals important patterns and relationships in the data. 
        Explore how different factors are connected to exploitation types.
        """)
        
        # Check if we have enough data for analysis
        if len(filtered_df) < 5:
            st.warning("Insufficient data for pattern analysis. Please adjust your filters to include more data.")
        else:
            relationships = [
                ('UN_COO_Region', 'exploitation_status', 'Region of Origin', 'Exploitation Type'),
                ('gender', 'exploitation_status', 'Gender', 'Exploitation Type'),
                ('UN_COE_Region', 'exploitation_status', 'Region of Exploitation', 'Exploitation Type')
            ]
            
            for var1, var2, name1, name2 in relationships:
                st.write("---")
                if all(col in filtered_df.columns for col in [var1, var2]):
                    if (filtered_df[var1].nunique() > 1 and 
                        filtered_df[var2].nunique() > 1):
                        results = calculate_chi_square_analysis(filtered_df, var1, var2)
                        display_refined_chi_square_results(results, name1, name2)
                    else:
                        st.warning(f"Insufficient variation in {name1} or {name2} for pattern analysis.")
                else:
                    st.warning(f"Required data for {name1} or {name2} not available.")
    
if __name__ == "__main__":
    main()