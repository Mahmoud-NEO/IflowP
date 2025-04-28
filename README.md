# Gas Well Deliverability Forecasting using a Physics-Informed Neural Network (PINN)

This repository contains a comprehensive project that aims to forecast gas well deliverability without relying on expensive well tests or advanced proprietary software. Instead, we develop a proxy model based on a Physics Informed Neural Network (PINN) that integrates empirical gas flow equations with data-driven modeling. The project emphasizes key concepts such as back pressure, log normalization, model architecture, and detailed result evaluation.

---

## Table of Contents

- [Introduction](#introduction)  
- [Equations and Theoretical Background](#equations-and-theoretical-background)  
- [Data Preprocessing and Log Normalization](#data-preprocessing-and-log-normalization)  
- [Feature Analysis and Importance](#feature-analysis-and-importance)  
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)  
- [PINN Model Building](#pinn-model-building)  
  - [Model Architecture](#model-architecture)  
  - [Physics-Informed Loss Function](#physics-informed-loss-function)  
  - [Training with LBFGS Optimizer](#training-with-lbfgs-optimizer)  
- [Results and Discussion](#results-and-discussion)  
- [Future Work and Next Steps](#future-work-and-next-steps)  
- [References](#references)  

---

## Introduction

Gas well deliverability forecasting is essential for planning and optimizing production. Traditional methods involve empirical correlations derived from well testing. However, these tests can be time-consuming and expensive. In this project, we develop a PINN that predicts the deliverability coefficients by leveraging both the underlying physics and an extensive dataset containing reservoir and fluid properties (e.g., porosity, permeability, gas formation volume factor). The model is designed to adhere to physical constraints, ensuring its predictions remain physically interpretable.

---

## Equation and Theoretical Background

### Back Pressure Equation (Equation 1)

The back pressure equation, introduced by Rawlins et al., is used to estimate the gas flow rate \(\,Q_g\) as follows:

$$
Q_g = C \\Bigl(\psi(P_r) - \psi(P_{wf})\Bigr)^n
$$

- **\(C\)**: Deliverability coefficient  
- **\(n\)**: Flow exponent  
- **\(\psi(P)\)**: A function that captures the pseudo-pressure term   

*The equations rely on constant coefficients typically derived from well testing. Since these tests may not always be feasible, our approach uses a PINN to predict these coefficients.*

---

## Data Preprocessing and Log Normalization

### Why and How?

- **Rationale:**  
  The original dataset includes measurements like gas flow rate (\(Q_g\)) and pressure drop (\(\Delta P\)). However, to better linearize the underlying physics (as seen in Equation 1), we apply a log transformation to these variables. This **log normalization** helps in stabilizing variance, reducing skewness, and highlighting multiplicative relationships.

- **Implementation:**  
  1. **Data Cleaning:** Rows with zero values in critical columns (`"Qg, mscf/d"` and `"Delta_P"`) are dropped to avoid issues in logarithmic transformation.  
  2. **Log Transformation:** We compute \(\log_{10}\) for both \(\Delta P\) and the gas flow rate \(Q_g\) (scaled appropriately) and store these in new columns (`Log_DP` and `Log_q`).

- **Observation:**  
  Post-normalization, the data distributions become more symmetric and are better suited for regression modeling.

- **Next Steps:**  
  Proceed with creating copies for analysis and visualizing the relationships (e.g., scatter plots of flow rate versus bottomhole pressure and histogram distributions).

---

## Feature Analysis and Importance

### Objectives

- **Understand Influences:**  
  Determine how each feature affects the gas flow rate.

- **Methods Used:**  
  - **Correlation Analysis:** A heatmap is generated to visualize pairwise correlations between features.  
  - **Lasso Regression:** Lasso is used for feature selection by imposing regularization, allowing us to rank features by importance.  
  - **ANOVA Test:** Selects the best predictors using statistical tests to rank features.

### Why

- **Rationale:**  
  By identifying the most influential features, we ensure that the model uses data that provide the most signal about gas flow behavior. This is particularly important because parameters like skin and drainage radius (\(R_e\), ft) are omitted to emulate a pre-test environment.

- **Observation:**  
  The analysis reveals that key features (such as permeability \(\,k\), pressure drop \(\,DP\), and thickness \(\,h\)) are highly correlated with gas flow rate as expected.

- **Next Steps:**  
  Use the insights from the feature ranking to select all available data (except those not available before well testing, such as skin and drainage radius) to train the predictive model.

---

## Principal Component Analysis (PCA)

### Why and How?

- **Rationale:**  
  With many potentially correlated features, PCA reduces dimensionality by transforming the data into a set of uncorrelated principal components. This simplifies the modeling task while retaining most of the variance.

- **Implementation:**  
  1. **Scaling:** Data is standardized using `StandardScaler`.  
  2. **PCA Transformation:** PCA is applied to capture 95th percentile of the variance, and the resulting principal components are saved for further modeling.  
  3. **Visualization:**  
     - A bar chart displays the percentage of variance explained by each component.  
     - Loadings plots show the contribution of original features to each principal component.  
     - A cumulative variance plot illustrates how many components are needed to capture the majority of the data variance.

- **Observation:**  
  The first principal component (PC1) typically explains the most variance, and the loadings help us interpret the physical meaning (e.g., fluid properties vs. petrophysical parameters).

- **Next Steps:**  
  The transformed data from PCA is used as input for the PINN model.

---

## PINN Model Building

### Model Architecture

- **Rationale:**  
  Instead of traditional empirical methods, a PINN offers a flexible and robust way to combine domain knowledge with deep learning. It allows us to predict key parameters such as **logC** and **n** (or alternatively, **a** and **b**) while enforcing the physics of gas flow.

- **Structure:**  
  - **Layers:** Four fully-connected layers with ReLU activation functions are used to capture non-linear relationships.  
  - **Outputs:**  
    - `output_logC`: Represents log_{10}(C), with no activation constraint.  
    - `output_n`: Represents the flow exponent \(n\), constrained to lie within a physically plausible range (transformed with a sigmoid function scaled to \([0.5, 5]\)).

- **Observation:**  
  The architecture is designed to balance complexity and interpretability, ensuring the outputs can be mapped back to physically meaningful parameters.

### Physics-Informed Loss Function

- **Why:**  
  Standard loss functions may not capture the underlying physics. The custom **physics_loss** compares the predicted log flow rate with a formulation derived from the physics (Equation 1 or 2), ensuring the network’s outputs remain consistent with known physical laws.

- **How:**  
  1. The network predicts log(C) and n.  
  2. The predicted gas flow rate is computed as:

  $$
  \log\bigl(q_{\text{pred}}\bigr) \;=\; \log(C) \;+\; n \,\times\, \log(\Delta P)
  $$

  3. A weighted mean squared error is computed, where the weights help stabilize training over different scales.

### Training with LBFGS Optimizer

- **Why LBFGS:**  
  LBFGS is a quasi-Newton method that uses second-order information to provide stable and informed parameter updates. It is particularly effective in full-batch scenarios and for loss landscapes governed by strong physical constraints.

- **How:**  
  1. Random seeds are fixed for reproducibility.  
  2. The training data (including PCA-transformed features and log-transformed targets) is scaled and split into training and testing sets.  
  3. The LBFGS optimizer is used with a closure function that:  
     - Sets the model to training mode.  
     - Computes the physics-informed loss.  
     - Performs backpropagation.  
  4. The training progress is logged over multiple epochs, and both training and testing losses are plotted for further analysis.

- **Observation:**  
  The loss curves (both overall and for epochs after the initial burn-in period) provide insights into convergence and potential overfitting issues.

- **Next Steps:**  
  After training, evaluate the model on both training and testing sets using various performance metrics.

---

## Results and Discussion

### Key Observations

1. **Error Metrics:**  
   - Metrics such as MSE, RMSE, MAE, and R² demonstrate that the PINN accurately captures the gas flow behavior.  
   - A high R² and low residual errors indicate that the model explains most of the variance in the data.

2. **Residual Analysis:**  
   - Residual plots for both training and testing data show random scatter around zero, with no systematic bias. This suggests the model generalizes well.  
   - Histograms of residuals reveal near-normal distributions, confirming that the errors are mostly small and random.

3. **Physical Consistency (Back Pressure):**  
   - The model's predictions obey the expected behavior: when pressure drawdown is near zero, the predicted flow rate is negligible.  
   - As pressure drawdown increases, the flow rate increases in a smooth, monotonic fashion-consistent with the Back Pressure Equation and gas flow physics.

4. **Parameter Interpretability:**  
   - The learned deliverability exponent \(n\) and coefficient \(C\) are in line with theoretical expectations.  
   - These outputs not only fit the data well but also provide meaningful physical insights, reinforcing the reliability of the model.

### Discussion

The PINN approach successfully combines data-driven learning with physics-based constraints. This hybrid strategy prevents the network from learning non-physical solutions and provides outputs that are directly interpretable in terms of known gas flow equations. While the model performs very well within the range of observed data, caution is advised when extrapolating to extreme conditions.

---

## Future Work and Next Steps

- **Hyperparameter Tuning:**  
  Further tuning of the neural network architecture and training parameters may yield even better performance.

- **Data Augmentation:**  
  Incorporating additional datasets or simulating different reservoir conditions can help improve generalizability.

- **Real-World Validation:**  
  Applying the model to new well data and comparing with real-time production figures will validate its robustness and practical utility.

- **Comples Cases:**  
  Consider integrating the relative permeability effects to account for the phase fluid flow.

---

## References

- Rawlins, E.L. and Schellhardt, M.A. 1935. Backpressure Data on Natural Gas Wells and Their Application to Production Practices, 7. Monograph Series, U.S. Bureau of Mines.
