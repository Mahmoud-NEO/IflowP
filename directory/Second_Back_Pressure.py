import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import os
import plotly.express as px
import plotly.graph_objects as go

from directory.GasProperties import GasProperties 

 # -----------------------------------------
# 0. Page Configuration & Theming
 # -----------------------------------------
st.set_page_config(
    page_title="Back Pressure IPR Prediction",
    page_icon="⛽",
    layout="wide"
)

# Custom CSS for styling 
st.markdown(
    """
    <style>
        /* Sidebar custom label */
        [data-testid="stSidebarNav"]::before {
            content: "IFP School";
            font-family: 'Comic Sans MS', sans-serif;
            margin-left: 100px;
            margin-top: 30px;
            font-size: 21px;
            position: relative;
            top: 5px;
            text-align: center;
            font-weight: bold;
        }
        /* Logo and title styling */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .logo-img {
            width: 120px;
            height: auto;
            margin-right: 10px;
        }
        .stTitle {
            font-size: 40px;
            font-weight: bold;
            color: #0256FE;
            margin-bottom: 20px;
            text-align: center;
            font-family: 'Comic Sans MS', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

 # -----------------------------------------
# 1. Logo & Title Display
 # -----------------------------------------
st.markdown(
    """
    <div class="logo-container">
        <img class="logo-img" src="https://upload.wikimedia.org/wikipedia/commons/6/69/IFP_Logo.png">
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='stTitle'>Back Pressure IPR Prediction</div>", unsafe_allow_html=True)

 # -----------------------------------------
# 2. Model Architecture
 # -----------------------------------------
class GasFlowModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output_logC = nn.Linear(64, 1)  # Predicts log10(C)
        self.output_n    = nn.Linear(64, 1)  # Predicts n

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        logC = self.output_logC(x)
        
        n = F.sigmoid(self.output_n(x)) * 0.5 + 0.5  

        return logC, n

 # -----------------------------------------
# 3. Caching: Load the PCA Model and PINN
 # -----------------------------------------
@st.cache_resource
def load_prediction_objects():
    base_dir = os.path.dirname(__file__)
    
    # Load PCA-related objects
    with open(os.path.join(base_dir, "scaler_pca.pkl"), "rb") as f:
        scaler_pca = pickle.load(f)
    with open(os.path.join(base_dir, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    
    # Load the scaler for PCA outputs
    with open(os.path.join(base_dir, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    
    # Load the trained PINN model
    model_path = os.path.join(base_dir, "second_pressure.pth")

    # Determine input dimension from scaler_X (number of features after PCA).
    input_dim = 6
    model = GasFlowModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return scaler_pca, pca, scaler_X, model

def predict_flow(df):
    # Load preprocessing objects and model
    scaler_pca, pca, scaler_X, model = load_prediction_objects()
    
    # Extract features for PCA transformation
    feature_cols = [
        "Pr, psi", "Temperature, F", "k, md", 
        "Porosity, fraction", "h, ft", "SG", 
        "Bg, bbl/scf", "Gas Viscosity, cp", "Gas Density, lb/ft3"
    ]
    X_raw = df[feature_cols].values

    # Apply the same transformations used in training
    X_scaled_for_pca = scaler_pca.transform(X_raw)
    X_pca = pca.transform(X_scaled_for_pca)
    X_model = scaler_X.transform(X_pca)
    
    # Convert to Torch tensor
    X_tensor = torch.tensor(X_model, dtype=torch.float32)
    
    model.eval()
    with torch.inference_mode():
        logC, n = model(X_tensor)
    
    # Compute real gas pseudo pressure for each row
    dp_list = []
    for _, row in df.iterrows():
        Pr = row["Pr, psi"]
        T  = row["Temperature, F"]
        SG = row["SG"]
        Pwf= row["Pwf, psi"]

        gas = GasProperties(gamma=SG, Pressure=Pr, Temperature=T)
        dp_val = gas.real_gas_pseudo_pressure(Pwf=Pwf)
        dp_list.append(dp_val)
    dp_array = np.array(dp_list)
    
    # Model outputs => logC + n*dp => log_q => Q
    logC_np = logC.squeeze().cpu().numpy()
    n_np    = n.squeeze().cpu().numpy()
    
    C_pred = 10 ** logC_np
    n_pred = n_np
    Q_pred = C_pred * (dp_array ** n_pred)
    
    return Q_pred, C_pred[-1], n_pred[-1], dp_array

 # -----------------------------------------
# 4. Sidebar Information
 # -----------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This tool uses a PINN model optimized for the **first back pressure equation** 
        to predict gas flow rate (q) based on various reservoir and fluid properties.
        
        **Instructions**  
        1. Go to the "Input" tab and set your reservoir & fluid properties.  
        2. Click "Predict".  
        3. Check the "Results & Confidence Interval" tab for outputs.  
        4. Explore anaylsis plot in "Distribution".
        """
    )

 # -----------------------------------------
# 5. Tabs for Input, Results, and Analysis
 # -----------------------------------------
tab1, tab2, tab3 = st.tabs(["Input", "Results & Confidence Interval", "Distribution"])

 # -----------------------------------------
# Tab 1: Input
 # -----------------------------------------
with tab1:
    st.subheader("Reservoir & Fluid Properties")
    pvt_available = st.toggle(
        "PVT Data Available", 
        value=False, 
        help="If enabled, you can provide your own Bg, viscosity, and density."
    )

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pr = st.number_input(
                'Reservoir Pressure, psi', 
                min_value=1000.0, max_value=10000.0, value=3000.0,
                help="Initial reservoir pressure in psi"
            )
            Tr = st.number_input(
                'Temperature, °F', 
                min_value=77.0, max_value=350.0, value=150.0,
                help="Reservoir temperature in Fahrenheit"
            )
        with col2:
            SG = st.number_input(
                'Gas Specific Gravity', 
                min_value=0.58, max_value=0.9, value=0.7,
                help="Gas specific gravity relative to air"
            )
            PHI = st.number_input(
                'Porosity (fraction)', 
                min_value=0.05, max_value=0.35, value=0.2,
                help="Formation porosity as fraction between 0.05 and 0.35"
            )
        with col3:
            PERM = st.number_input(
                'Permeability, md', 
                min_value=1.0, max_value=1000.0, value=100.0,
                help="Formation permeability in millidarcies"
            )
            h = st.number_input(
                'Net Thickness, ft', 
                min_value=5.0, max_value=300.0, value=50.0,
                help="Net pay thickness in feet"
            )
        
        if pvt_available:
            with st.expander("PVT Data"):
                Bg = st.number_input(
                    'Bg, bbl/scf', 
                    min_value=0.0001, max_value=0.01, 
                    value=0.0015, format="%.6f",
                    help="Gas formation volume factor (bbl/scf)"
                )
                Viscosity = st.number_input(
                    'Gas Viscosity, cp', 
                    min_value=0.001, max_value=0.09, 
                    value=0.02, format="%.4f",
                    help="Gas viscosity in centipoise"
                )
                rho = st.number_input(
                    'Gas Density, lb/ft³', 
                    min_value=5.0, max_value=20.0, 
                    value=10.0,
                    help="Gas density in lb/ft³"
                )
        else:
            st.info("The gas properties will be estimated by correlations!")
            gas_est = GasProperties(gamma=SG, Pressure=Pr, Temperature=Tr)
            Bg = gas_est.Bg()
            Viscosity = gas_est.viscosity_gas()
            rho = gas_est.gas_density()
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Generate a range of Pwf values from Pr down
        Pwf_vals = np.arange(Pr, 14.7 - 1, -100)
        input_df = pd.DataFrame({
            "Pr, psi": np.full(len(Pwf_vals), Pr),
            "Temperature, F": np.full(len(Pwf_vals), Tr),
            "k, md": np.full(len(Pwf_vals), PERM),
            "Porosity, fraction": np.full(len(Pwf_vals), PHI),
            "h, ft": np.full(len(Pwf_vals), h),
            "SG": np.full(len(Pwf_vals), SG),
            "Bg, bbl/scf": np.full(len(Pwf_vals), Bg),
            "Gas Viscosity, cp": np.full(len(Pwf_vals), Viscosity),
            "Gas Density, lb/ft3": np.full(len(Pwf_vals), rho),
            "Pwf, psi": Pwf_vals
        })

        with st.status("Predicting...", expanded=True) as status:
            Q_pred, C_pred, n_pred, dp = predict_flow(input_df)
            # Save results to session_state
            st.session_state["predicted_df"] = input_df.copy()
            st.session_state["predicted_df"]["Qg, Mscf/d"] = Q_pred / 1000
            st.session_state["predicted_df"]["DP"] = dp
            st.session_state["C_pred"] = C_pred
            st.session_state["n_pred"] = n_pred
            time.sleep(1)
            status.update(label="Done!", state="complete", expanded=False)

        st.success("Switch to 'Results & Confidence Interval' tab to see predictions.")

 # -----------------------------------------
# Tab 2: Results & Confidence Interval
 # -----------------------------------------
with tab2:
    st.subheader("Predicted Results with Confidence Interval")

    if "predicted_df" not in st.session_state:
        st.warning("No predictions yet. Please go to the 'Input' tab and click 'Predict'.")

    else:
        df_pred = st.session_state["predicted_df"]
        c_val = st.session_state["C_pred"]
        n_val = st.session_state["n_pred"]

        # Display metrics side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Predicted Gas AOF (Mscf/d)",
                value=f"{round(df_pred['Qg, Mscf/d'].iloc[-1], 0):.0f}"
            )
        with col2:
            st.metric(
                label="Predicted C",
                value=f"{c_val:.3f}"
            )
        with col3:
            st.metric(
                label="Predicted n",
                value=f"{n_val:.4f}"
            )

        st.dataframe(df_pred)

        # Confidence interval plot
        mae = 1086  # Estimated MAE for back pressure fitting
        Pwf_vals = df_pred["Pwf, psi"].tolist()
        Qg_vals  = df_pred["Qg, Mscf/d"].tolist()

        Qg_lower = [max(0, q - mae) for q in Qg_vals]
        Qg_upper = [q + mae for q in Qg_vals]

        fig = go.Figure()

        # Predicted Qg
        fig.add_trace(go.Scatter(
            x=Qg_vals,
            y=Pwf_vals,
            mode='lines+markers',
            name='Predicted Qg',
            line=dict(color='blue', width=2)
        ))

        # Upper bound
        fig.add_trace(go.Scatter(
            x=Qg_upper,
            y=Pwf_vals,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 1)', width=1),
            showlegend=False
        ))

        # Lower bound + fill
        fig.add_trace(go.Scatter(
            x=Qg_lower,
            y=Pwf_vals,
            mode='lines',
            fill='tonextx',
            fillcolor='rgba(255, 0, 0, 0.4)',
            line=dict(color='rgba(255, 0, 0, 1)', width=1),
            name='MAE Confidence Interval'
        ))

        fig.update_layout(
            xaxis=dict(
                title=dict(text='Qg, MSCF/D', font=dict(size=20)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='Pwf, psi', font=dict(size=20)),
                tickfont=dict(size=14)
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.update_xaxes(range=[0, max(Qg_vals)+1000])
        fig.update_yaxes(range=[0, max(Pwf_vals)])

        st.plotly_chart(fig, use_container_width=True)

        # CSV Download
        csv_data = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv"
        )

 # -----------------------------------------
# Tab 3: Distribution Visualization
 # -----------------------------------------
with tab3:
    if "predicted_df" not in st.session_state:
        st.warning("No predictions yet. Please go to the 'Input' tab and click 'Predict'.")

    else:
        df_pred = st.session_state["predicted_df"]

        # -----------------------------------------
        # 3.A Dynamic bin slider for Qg histogram
         # -----------------------------------------
        st.write("### Distribution of Predicted Flow Rates")
        bin_count = st.slider("Number of bins", min_value=5, max_value=50, value=10, step=1)
        
        fig_hist = px.histogram(
            df_pred,
            x="Qg, Mscf/d",
            nbins=bin_count
        )
        fig_hist.update_layout(
            xaxis_title=dict(text="Qg, Mscf/d", font=dict(size=20)),
            yaxis_title=dict(text="Count", font=dict(size=20)),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    