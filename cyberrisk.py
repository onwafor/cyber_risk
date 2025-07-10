import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
from PIL import Image  # Python Imaging Library

# Check for XGBoost and install if needed
try:
    from xgboost import XGBRegressor
except ImportError:
    st.warning("XGBoost not found. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBRegressor

# Load your image
# image = Image.open("cedaf_ai2.png")  # or .png, .gif, etc.

st.set_page_config(layout="wide")

# Load Model
with open('xgb_model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Load Model metadata
with open('model_metadata2.pkl', 'rb') as file:
    metadata = pickle.load(file)



@st.cache_data  # Cache the data loading and calculations
def load_and_process_data():
    # Load your dataset (update path as needed)
    df = pd.read_excel("Cyber Risk Data.xlsx")
    
    # Compute expected annual and residual loss
    df['Expected Annual Loss ($)'] = (
        df['Frequency'] *
        df['Vulnerability Score'] *
        (df['Primary Loss ($)'] + df['Secondary Loss ($)'])
    )
    
    df['Residual Loss ($)'] = (
        df['Expected Annual Loss ($)'] *
        (1 - df['Control Maturity Score'])
    )
    
    # Monte Carlo CVaR Simulation
    n_simulations = 10000
    simulated_means = []
    simulated_cvars = []

    for _, row in df.iterrows():
        lambda_freq = row['Frequency']
        residual_loss = row['Residual Loss ($)']
        
        if lambda_freq == 0 or residual_loss == 0:
            simulated_means.append(0.0)
            simulated_cvars.append(0.0)
        else:
            avg_loss = residual_loss / lambda_freq
            mu = np.log(avg_loss + 1e-6)
            sigma = 0.8
            
            freq_sim = np.random.poisson(lam=lambda_freq, size=n_simulations)
            loss_per_event = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)
            total_loss = freq_sim * loss_per_event
            
            simulated_means.append(np.mean(total_loss))
            tail = total_loss[total_loss >= np.percentile(total_loss, 95)]
            simulated_cvars.append(np.mean(tail))

    df['Simulated Mean Loss ($)'] = simulated_means
    df['Simulated CVaR 95% ($)'] = simulated_cvars
    
    return df

df = load_and_process_data()


def main():
    # st.title("Cyber Risk Insurance Executive Dashboard")
    #st.write("Cyber Risk Quantification Model")

    st.markdown("<h1 style='text-align: center; font-size: 30px;color: #002a6f ;'> Cyber Risk Intelligence</h1>", 
    unsafe_allow_html=True )
    st.markdown("<p style='text-align: center; font-size: 25px;'>Predictive and Scenario Analysis Dashboard </p>", 
    unsafe_allow_html=True )
    #st.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)


    # Display in sidebar
    #st.sidebar.image(image)
    #st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.header("Filter Data")
    st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)

    # Get unique values for filters
    all_industries = ["All"] + sorted(df["Industry"].unique().tolist())
    all_vectors = ["All"] + sorted(df["Attack Vector"].unique().tolist())
    
    # Create dropdown filters
    selected_industry = st.sidebar.selectbox("Select Industry", all_industries)
    st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)
    selected_attack = st.sidebar.selectbox("Select Attack Vector", all_vectors)
    st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)
   
    # Create frequency slider
    min_freq, max_freq = int(df['Frequency'].min()), int(df['Frequency'].max())
    freq_range = st.sidebar.slider(
        "Select Frequency Range",
        min_value=min_freq,
        max_value=max_freq,
        value=(min_freq, max_freq)
    )
    
    st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)



    # Filter data based on selections
    filtered_df = df.copy()
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry"] == selected_industry]
    if selected_attack != "All":
        filtered_df = filtered_df[filtered_df["Attack Vector"] == selected_attack]
    
    # Apply frequency filter
    filtered_df = filtered_df[
        (filtered_df['Frequency'] >= freq_range[0]) & 
        (filtered_df['Frequency'] <= freq_range[1])
    ]
    
    # Display filtered data stats
    st.sidebar.write(f" {len(filtered_df)} records matching your filters")
    #st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)
    
    # Create dashboard visualizations
    create_dashboard_visualizations(filtered_df)


def create_dashboard_visualizations(df):
    # Set style
    sns.set(style="whitegrid")
    
    # Determine top industries and vectors based on filtered data
    top_industries = df.groupby('Industry')['Simulated CVaR 95% ($)'].mean().nlargest(7).index
    df_top_industry = df[df['Industry'].isin(top_industries)]
    
    top_vectors = df.groupby('Attack Vector')['Simulated CVaR 95% ($)'].mean().nlargest(7).index
    df_top_vector = df[df['Attack Vector'].isin(top_vectors)]
    
    # Create tabs for different visualizations
    tab2, tab3, tab5 = st.tabs([ "Exposure Analysis", "CVaR Heatmap", "PREDICTIVE ANALYTICS"])
    
    with tab2:
        # Exposure analysis charts
        col1, col2 = st.columns(2)
        plt.style.use('dark_background')
        with col1:
            #st.subheader("CVaR vs Risk Exposure by Industry")
            plt.style.use('dark_background')
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>CVaR vs Risk Exposure by Industry</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_top_industry, x='Risk Exposure Score', y='Simulated CVaR 95% ($)', 
                           hue='Industry', alpha=0.8, ax=ax)
            ax.set_xlabel("Risk Exposure Score")
            ax.set_ylabel("Simulated CVaR 95% ($)")
            st.pyplot(fig)
        
        with col2:
            #st.subheader("CVaR vs Risk Exposure by Attack Vector")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>CVaR vs Risk Exposure by Attack Vector</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_top_vector, x='Risk Exposure Score', y='Simulated CVaR 95% ($)', 
                           hue='Attack Vector', alpha=0.8, ax=ax)
            ax.set_xlabel("Risk Exposure Score")
            ax.set_ylabel("Simulated CVaR 95% ($)")
            st.pyplot(fig)
    
    with tab3:
        # Heatmap
        #st.subheader("CVaR Heatmap by Industry × Attack Vector")
        st.markdown( "<h1 style='font-size: 16px; text-align: center;'>CVaR Heatmap by Industry × Attack Vector</h1>", 
            unsafe_allow_html=True )
        pivot = df.pivot_table(values='Simulated CVaR 95% ($)', index='Industry', columns='Attack Vector', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax, annot_kws={"fontsize": 8})
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        st.pyplot(fig)

    with tab5:

        input_data = {}
        # Get unique values from original data
        all_industries = sorted(df["Industry"].unique().tolist())
        all_vectors = sorted(df["Attack Vector"].unique().tolist())
        all_assets_at_risk = sorted(df["Asset at Risk"].unique().tolist())
        
        coli1, coli2, coli3, coli4, coli5, coli6 = st.columns(6)
        with coli1:
            industry_selection = st.selectbox("Select Industry", all_industries)
            input_data['Frequency'] = st.slider("Frequency (number of incidents)", min_value=1, max_value=20, value=10)
        with coli2:
            attack_vector_selection = st.selectbox("Attack Vector", all_vectors)
            input_data['Vulnerability Score'] = st.slider("Vulnerability Score (0-1)", min_value=0.1, max_value=1.0, value=0.1 )
        with coli3:
            asset_at_risk_selection = st.selectbox("Asset at Risk", all_assets_at_risk)
            input_data['Control Maturity Score'] = st.slider( "Control Maturity Score (0-1)", min_value=0.0, max_value=1.0, value=0.1 )
        with coli4:
            input_data['Primary Loss ($)'] = st.number_input("Primary Loss ($)", min_value=0, value=500000, step=10000)
            input_data['Downtime (hrs)'] = st.slider( "Downtime (hours)", min_value=1, max_value=120, value=30 )
        with coli5:
            input_data['Secondary Loss ($)'] = st.number_input("Secondary Loss ($)", min_value=0, value=100000, step=10000)
            input_data['Risk Exposure Score'] = st.slider("Risk Exposure Score (0-1)", min_value=0.0, max_value=1.0, value=0.35, step=0.01 )
        with coli6:
            input_data['Industry'] = metadata['encoders']['Industry'].transform([industry_selection])[0]
            input_data['Attack Vector'] = metadata['encoders']['Attack Vector'].transform([attack_vector_selection])[0]
            input_data['Asset at Risk'] = metadata['encoders']['Asset at Risk'].transform([asset_at_risk_selection])[0]

            # Convert input to DataFrame in correct feature order
            input_df = pd.DataFrame([input_data], columns=metadata['features'])

            # Ensure all columns are numeric
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='raise')

            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display prediction
            #st.write("#### Prediction Result")
            st.success(f"Expected Annual Loss: **${prediction:,.2f}**")

            # Display calculated total loss
            input_data['Total Loss ($)'] = input_data['Primary Loss ($)'] + input_data['Secondary Loss ($)']
            st.markdown(f"**Total Loss:** ${input_data['Total Loss ($)']:,.2f}")
       # st.divider()



        # Filter data based on selections
        filtered_df2 = df.copy()
        if industry_selection != "All":
            filtered_df2 = filtered_df2[filtered_df2["Industry"] == industry_selection]
        if attack_vector_selection != "All":
            filtered_df2 = filtered_df2[filtered_df2["Attack Vector"] == attack_vector_selection]
        
        # Apply frequency filter
        filtered_df2 = filtered_df2[
            (filtered_df2['Frequency'] >= 0) & 
            (filtered_df2['Frequency'] <= 1)
        ]
        

        # Create dashboard visualizations
        #create_dashboard_visualizations2(filtered_df2)


        def create_dashboard_visualizations2(df):
            # Set style
            
            # Determine top industries and vectors based on filtered data
            top_industries = df.groupby('Industry')['Simulated CVaR 95% ($)'].mean().nlargest(7).index
            df_top_industry = df[df['Industry'].isin(top_industries)]
            
            top_vectors = df.groupby('Attack Vector')['Simulated CVaR 95% ($)'].mean().nlargest(7).index
            df_top_vector = df[df['Attack Vector'].isin(top_vectors)]

        col2a, col2b, col2c= st.columns(3)
        with col2a:
            plt.style.use('dark_background')
            #st.subheader("Top Industries by Average CVaR")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>Top Industries by Average CVaR</h1>", 
            unsafe_allow_html=True )
            industry_avg_cvar = df_top_industry.groupby('Industry')['Simulated CVaR 95% ($)'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x=industry_avg_cvar.values, y=industry_avg_cvar.index, palette='Reds_r', ax=ax)
            ax.set_xlabel("Average CVaR 95% ($)")
            st.pyplot(fig)

        
        with col2b:
            #st.subheader("Top Attack Vectors by Average CVaR")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>Top Attack Vectors by Average CVaR</h1>", 
            unsafe_allow_html=True )
            vector_avg_cvar = df_top_vector.groupby('Attack Vector')['Simulated CVaR 95% ($)'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x=vector_avg_cvar.values, y=vector_avg_cvar.index, palette='Oranges_r', ax=ax)
            ax.set_xlabel("Average CVaR 95% ($)")
            st.pyplot(fig)

        with col2c:
            plt.style.use('dark_background')
            # Initialize lists for outputs
            simulated_means = []
            simulated_cvars = []

            # Number of simulations
            n_simulations = 10000
            portfolio_annual_losses = []

            # Loop through each record
            for idx, row in df.iterrows():
                lambda_freq = row['Frequency']
                residual_loss = row['Residual Loss ($)']
                
                # If frequency is zero, assume zero expected loss
                if lambda_freq == 0 or residual_loss == 0:
                    simulated_means.append(0.0)
                    simulated_cvars.append(0.0)
                else:
                    # Estimate per-event loss for lognormal sampling
                    avg_loss_per_event = residual_loss / lambda_freq
                    mu = np.log(avg_loss_per_event + 1e-6)
                    sigma = 0.8  # can be tuned
                    
                    # Simulate frequency
                    freq_sim = np.random.poisson(lam=lambda_freq, size=n_simulations)
    
                    # Simulate per-event loss
                    loss_per_event = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)
                    
                    # Total simulated loss = frequency × severity
                    total_loss = freq_sim * loss_per_event
                    
                    # Store statistics
                    simulated_means.append(np.mean(total_loss))
                    tail_losses = total_loss[total_loss >= np.percentile(total_loss, 95)]
                    simulated_cvars.append(np.mean(tail_losses))

            # Add results to DataFrame
            df['Simulated Mean Loss ($)'] = simulated_means
            df['Simulated CVaR 95% ($)'] = simulated_cvars
            # For each simulation, sample from the simulated mean losses of all records
            for _ in range(n_simulations):
                # Random number of events per year: based on total expected frequency
                total_events = np.random.poisson(lam=df['Frequency'].sum())
                
                if total_events == 0:
                    portfolio_annual_losses.append(0.0)
                else:
                    # Randomly sample 'total_events' losses from the Simulated Mean Losses
                    sampled_losses = np.random.choice(df['Simulated Mean Loss ($)'].values, size=total_events, replace=True)
                    portfolio_annual_losses.append(np.sum(sampled_losses))

            # Convert to numpy array for further analysis
            portfolio_annual_losses = np.array(portfolio_annual_losses)

            # Preview: display summary statistics
            #import matplotlib.pyplot as plt
            st.markdown( "<h1 style='font-size: 16px;text-align: center;'>Simulated Portfolio Annual Loss Distribution</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(8, 2.7))
            ax.hist(portfolio_annual_losses, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(portfolio_annual_losses), color='red', linestyle='dashed', label='Mean Loss')
            ax.axvline(np.percentile(portfolio_annual_losses, 95), color='darkorange', linestyle='dashed', label='95% VaR')
            ax.axvline(np.mean(portfolio_annual_losses[portfolio_annual_losses > np.percentile(portfolio_annual_losses, 95)]), 
                        color='purple', linestyle='dashed', label='95% CVaR')
            ax.set_xlabel("Total Annual Loss ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)


        
        col3a, col3b= st.columns([0.6,0.4])
             
        with col3a:
            # Create SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)       
        
            # Waterfall plot
            st.markdown( "<h1 style='font-size: 16px;text-align: center;'>How Each Factor Affects This Prediction</h1>", 
            unsafe_allow_html=True )

            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)

            # Create controlled figure and force SHAP to use it
            fig, ax = plt.subplots(figsize=(5, 3))
            shap.plots.waterfall(shap_values[0], max_display=20, show=False)

            # Force grid lines
            ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.4)
            ax.grid(False, axis='y')

            st.pyplot(fig)
            plt.close(fig)
                   

        with col3b:
            plt.style.use('dark_background')
            # Create quantile-based bins for Risk Exposure Score
            df['Risk Tier'] = pd.qcut(
            df['Risk Exposure Score'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            # Map tiers to colors
            tier_colors = { 'High Risk': 'red', 'Medium Risk': 'gold', 'Low Risk': 'green' }
            df['Color'] = df['Risk Tier'].map(tier_colors)

            st.markdown( "<h1 style='font-size: 16px;text-align: center;'>CVaR vs Residual Risk (Bubble Size = Exposure Score)</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(4, 6))
            ax.scatter(
                    x=df['Residual Loss ($)'],
                    y=df['Simulated CVaR 95% ($)'],
                    s=df['Risk Exposure Score'] * 1000,  # Scale bubble size
                    c=df['Color'],
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.5
                )

            # Add quadrant lines
            x_median = df['Residual Loss ($)'].median()
            y_median = df['Simulated CVaR 95% ($)'].median()
            ax.axvline(x=x_median, color='gray', linestyle='--')
            ax.axhline(y=y_median, color='gray', linestyle='--')

            # Labels and title
            ax.set_xlabel("Residual Loss ($)")
            ax.set_ylabel("Simulated CVaR 95% ($)")

            # Create manual legend
            import matplotlib.patches as mpatches
            legend_handles = [
            mpatches.Patch(color='red', label='High Risk'),
            mpatches.Patch(color='gold', label='Medium Risk'),
            mpatches.Patch(color='green', label='Low Risk')
            ]
            ax.legend(handles=legend_handles, title="Risk Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            

    st.divider()   




                
    
if __name__ == "__main__":
    main()