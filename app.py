import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config
st.set_page_config(page_title="Water Usage Dashboard", page_icon="💧", layout="wide")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_excel('dataset/data.xlsx')  # Replace with your actual file path
    return df

df = load_data()
category_columns = ['IMD', 'CACI', 'TARIFF', 'AGE', 'OCCUPANCY', 'CUSTOMER_EMPLOYMENT_STATUS', 'CUSTOMER_HOME_OWNER_STATUS']

# Clean column names
df.columns = df.columns.str.strip()

# Sidebar
st.sidebar.image("image/logo.png", use_column_width=True)
st.sidebar.title('Navigation')

# Move section selection to sidebar
section = st.sidebar.radio(
    "Choose a section",
    ["Overview", "Missing Data Analysis", "Detailed Analysis", "Advanced Visualizations"]
)

# Main page
st.title('Water Usage Dashboard')

# Content based on sidebar selection
if section == "Overview":
    st.header('Key Statistics')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}", help="Total number of records in the dataset")
        st.metric("Average PCC", f"{df['PCC'].mean():.2f}", help="Overall average Per Capita Consumption")

    with col2:
        st.metric("Unique IMD Categories", df['IMD'].nunique(), help="Number of unique IMD (Index of Multiple Deprivation) categories")
        st.metric("Unique CACI Categories", df['CACI'].nunique(), help="Number of unique CACI (Consumer Classification) categories")

    with col3:
        st.metric("Average Occupancy", f"{df['OCCUPANCY'].mean():.2f}", help="Average number of occupants per household")
        st.metric("Most Common Employment Status", df['CUSTOMER_EMPLOYMENT_STATUS'].mode()[0], help="Most frequently occurring employment status")

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Top 5 IMD Categories")
        st.write(df['IMD'].value_counts().head())

    with col5:
        st.subheader("Top 5 CACI Categories")
        st.write(df['CACI'].value_counts().head())

elif section == "Missing Data Analysis":
    st.header('Missing Data Analysis')
    
    missing_values = df.isnull().sum()
    missing_percentages = 100 * df.isnull().sum() / len(df)
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage Missing': missing_percentages
    })
    
    st.subheader("Missing Values Summary")
    st.write(missing_df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_df['Percentage Missing'].plot(kind='bar', ax=ax)
    plt.title('Percentage of Missing Values by Feature', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Percentage Missing', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(missing_df['Percentage Missing']):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    st.pyplot(fig)

elif section == "Detailed Analysis":
    st.header('Detailed Analysis')
    
    category_columns = ['IMD', 'CACI', 'TARIFF', 'AGE', 'OCCUPANCY', 'CUSTOMER_EMPLOYMENT_STATUS', 'CUSTOMER_HOME_OWNER_STATUS']
    selected_category = st.selectbox('Select Category for Analysis', category_columns)
    
    # Box plot
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(x=selected_category, y='PCC', data=df, ax=ax)
    plt.title(f'Water Consumption by {selected_category}', fontsize=20)
    plt.xlabel(selected_category, fontsize=14)
    plt.ylabel('PCC (Water Consumption)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    medians = df.groupby(selected_category)['PCC'].median()
    vertical_offset = df['PCC'].median() * 0.05
    for i, tick in enumerate(ax.get_xticklabels()):
        category = tick.get_text()
        if category in medians.index:
            ax.text(i, medians[category] + vertical_offset, f'Median: {medians[category]:.0f}', 
                    horizontalalignment='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Bar chart
    avg_pcc_by_category = df.groupby(selected_category)['PCC'].mean().reset_index()
    fig_bar = px.bar(avg_pcc_by_category, x=selected_category, y='PCC', title=f'Average PCC by {selected_category}')
    st.plotly_chart(fig_bar)

    # Distribution of PCC
    st.subheader("Distribution of Water Consumption (PCC)")
    fig_dist = px.histogram(df, x='PCC', nbins=50, title='Distribution of PCC')
    st.plotly_chart(fig_dist)


elif section == "Advanced Visualizations":
    st.header('Advanced Visualizations')
    
    # # Correlation Heatmap
    # st.subheader("Correlation Heatmap")
    # numeric_cols = ['PCC', 'IMD', 'AGE', 'OCCUPANCY']
    # corr = df[numeric_cols].corr()
    # fig_heatmap = ff.create_annotated_heatmap(
    #     z=corr.values,
    #     x=list(corr.columns),
    #     y=list(corr.index),
    #     annotation_text=corr.round(2).values,
    #     showscale=True)
    # st.plotly_chart(fig_heatmap)
#######################################################################################################
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    import plotly.figure_factory as ff
    import streamlit as st

    # Assume 'df' is your dataframe
    numeric_cols = ['PCC', 'AGE', 'OCCUPANCY']
    categorical_cols = ['CUSTOMER_EMPLOYMENT_STATUS']

    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = pd.DataFrame(
        encoder.fit_transform(df[categorical_cols]), 
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Combine numeric and encoded categorical data
    combined_data = pd.concat([df[numeric_cols], encoded_cats], axis=1)

    # Calculate correlation
    corr = combined_data.corr()

    # Sort correlation matrix to ensure diagonal alignment
    corr = corr.reindex(index=corr.columns, columns=corr.columns)

    # Create heatmap
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        showscale=True
    )

    # Update the layout to make the heatmap square
    fig_heatmap.update_layout(
        width=1000,
        height=1000,
        title="Correlation Heatmap"
    )

    # Display the heatmap
    st.plotly_chart(fig_heatmap)
###################################################################################################3
    # Scatter plots
    st.subheader("Scatter Plots")
    scatter_x = st.selectbox("Select X-axis for Scatter Plot", ['IMD', 'AGE', 'OCCUPANCY'])
    fig_scatter = px.scatter(df, x=scatter_x, y='PCC', color='CACI',
                             hover_data=['CUSTOMER_EMPLOYMENT_STATUS', 'CUSTOMER_HOME_OWNER_STATUS'])
    st.plotly_chart(fig_scatter)

    # Multi-Column Analysis
# In the Advanced Visualizations section, update the Multi-Column Analysis part:

    st.subheader('Multi-Column Analysis')
    selected_columns = st.multiselect('Select Columns for Grouped Analysis', category_columns)

    if selected_columns:
        if len(selected_columns) == 1:
            # Aggregate data
            grouped_df = df.groupby(selected_columns[0]).agg({
                'PCC': 'mean',
                'ID': lambda x: ', '.join(x.astype(str))  # Join all IDs in the group
            }).reset_index()
            
            fig = px.bar(grouped_df, x=selected_columns[0], y='PCC',
                        title=f'Average PCC by {selected_columns[0]}',
                        hover_data=['ID'])  # Include IDs in hover data
            
        elif len(selected_columns) == 2:
            # Aggregate data
            grouped_df = df.groupby(selected_columns).agg({
                'PCC': 'mean',
                'ID': lambda x: ', '.join(x.astype(str))  # Join all IDs in the group
            }).reset_index()
            
            fig = px.bar(grouped_df, x=selected_columns[0], y='PCC', color=selected_columns[1],
                        title=f'Average PCC by {selected_columns[0]} and {selected_columns[1]}',
                        hover_data=['ID'])  # Include IDs in hover data
            
        else:
            st.write("Please select 1 or 2 columns for visualization.")
            st.write(df.groupby(selected_columns)['PCC'].mean().reset_index())
        
        if 'fig' in locals():
            fig.update_layout(height=600, barmode='group')  # Use 'group' for grouped bar chart
            st.plotly_chart(fig, use_container_width=True)

        # Add a section to display potential outliers
        st.subheader("Potential Outliers")
        outlier_threshold = df['PCC'].quantile(0.95)  # Adjust this threshold as needed
        outliers = df[df['PCC'] > outlier_threshold].sort_values('PCC', ascending=False)
        
        st.write(f"Showing customers with PCC above the 95th percentile ({outlier_threshold:.2f}):")
        st.write(outliers[['ID', 'PCC'] + selected_columns])
    # Feature Importance (placeholder - uncomment and adjust if you have this data)
    # st.subheader("Feature Importance")
    # importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    # importance_df = importance_df.sort_values('importance', ascending=False).head(10)
    # fig_importance = px.bar(importance_df, x='importance', y='feature', orientation='h', 
    #                         title='Top 10 Most Important Features')
    # st.plotly_chart(fig_importance)

# Data download option
csv = df.to_csv().encode('utf-8')
st.sidebar.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='water_usage_data.csv',
    mime='text/csv',
)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ by United Utilities")


# to run the app locally run following command 
# 'streamlit run app.py'    