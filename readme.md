# Region I directly written! A component-based interactive adaptive multi-scale two-dimensional high-innovation time series prediction network based on STL decomposition

\tPreface
In response to the issues of strong non-stationarity in transformer oil temperature sequences, obvious periodic fluctuations, and complex multivariate coupling, this issue proposes a high-innovation time series prediction model.
\tSTLCIAM2DNet:   
\tSTL-based Component-Interactive Adaptive Multi-scale 2D Temporal Network  
\t  
\t(Adaptive Multi-Scale Two-Dimensional Temporal Prediction Network Based on STL Decomposition)
The model first utilizes the STL decomposition method to split the original multivariate time series into trend components, seasonal components, and residual components. Subsequently, adaptive multi-scale two-dimensional convolution encoders are constructed for different time characteristics. Furthermore, the dynamic relationship between trends, seasons and residuals is explicitly modeled through the inter-component interaction module. Finally, the future transformer oil temperature prediction results are obtained through component-adaptive fusion and direct prediction of the head output.
We also provide detailed materials, instructional documents, and video tutorials, including how to replace your own dataset, parameter tuning tutorials, etc. The code is commented line by line, with detailed introductions of parameters:
● Datasets: Power data set, wind power data set of a wind farm, wind speed dataset, etc.
● Environment Framework: Python 3.11, PyTorch 2.1 and above can be run.
● Model score: 0.95 on test set
● Usage Object: Paper requirements, graduation design requirements.
● Code Guarantee: The code comments are detailed and ready to run.
1. Introduction to the innovative model
1.1 Overall Process of the Model
\t  
The overall prediction process of STL-CIAM2DNet is as follows:
Original multivariate time series
\t→ Data preprocessing and normalization
\t→ STL decomposition
Trend term, seasonal term, residual term
\t→ Adaptive Multi-Scale Two-Dimensional Convolution Coding with Components
\t→ Interaction modeling between components
\t→ Component-adaptive fusion
\t→ Two-dimensional Temporal Decoding
\t→ Transformer oil temperature prediction output
The model first sorts the transformer operation data by time, interpolates missing values, selects features, and normalizes them. Subsequently, it performs STL decomposition on continuous time series including target oil temperature and related load variables to obtain trend components, seasonal components, and residual components. These three components are then fed into corresponding branch encoders to extract joint temporal-variable features at different scales. Subsequently, the model learns the mutual influence between these components through a component interaction module, and generates a unified time-series representation through a fusion module. Finally, the model outputs future transformer oil temperature predictions.
1.2 Model innovation points
1. Innovation Point: STL decomposition-driven component-level modeling framework
\tSTLCIAM2DNet first decomposes the original multivariate time series into:
Trend: Used to describe the long-term changes in the temperature of transformer oil and related variables.
\tSeasonal: Used to describe the periodic changes in oil temperature sequences, such as daily, weekly or load operating cycles.
\tResidual: Used to depict short-term disturbances, sudden fluctuations, and local anomalies that are difficult to explain by trends and cycles.
Through STL decomposition, the model can disassemble complex non-stationary sequences into multiple components with clearer structures and stronger predictability. This approach no longer directly confronts the mixed raw sequence but instead learns more targeted patterns in different component spaces. For transformer oil temperature prediction, this decomposition holds significant meaning: the trend term reflects the long-term evolution of the equipment's thermal state, the seasonal term represents operational cycles and load patterns, while the residual term captures short-term disturbances and anomalies. Together, these three components form a complete description of the oil temperature change process.
2. Innovation Point: Adaptive Multi-Scale Two-Dimensional Convolution Coder
1. Different STL components have different temporal characteristics, thus they are not suitable for unified modeling using the same network structure.
STLCIAM2DNet is designed with adaptive multi-scale two-dimensional convolution encoders for trend, seasonal, and residual components respectively:
The trend branch focuses on longer time scales to capture the long-term smooth changes in oil temperature;
The seasonal branch focuses on periodic scales to capture intra-day, weekly, etc., cyclic fluctuations;
The residual branch focuses on short time scales to capture sudden disturbances and local anomalies.
Unlike ordinary one-dimensional TCNs, STLCIAM2DNet uses two-dimensional temporal convolutions while extracting features in both the time and variable dimensions. The time-axis convolution is used to model historical sequence dependencies, while the variable-axis convolution is used to learn the local coupling relationships between oil temperature, load, and other operational variables.
Additionally, multiple convolution scales are set within each branch, and the scale attention mechanism adaptively fuses features from different scales. The model can automatically determine which short-term changes, mid-term fluctuations, or long-term trends should be paid more attention to based on the current sample features, thereby enhancing its ability to adapt to complex oil temperature change patterns.
Innovation Point 3: Interaction Modeling Module Between Components
Traditional decomposition prediction models typically model the trend, seasonal, and residual components separately and then directly concatenate them. However, this approach ignores the dynamic correlations that may exist between different components.
In fact, the changes in transformer oil temperature are not completely independent of each other. For example, a long-term increase in load might affect the amplitude of the oil temperature cycle fluctuations; periodic load changes might influence the distribution of residual disturbances; sudden residual fluctuations might also predict subsequent trend changes.
STLCIAM2DNet has designed a Component Interaction Module, CIM, to explicitly learn the interdependencies between trend components, seasonal components, and residual components through multi-head self-attention mechanisms. After component interaction, the model generates corresponding gating weights that inversely modulate the features of each component. This means that the model not only learns to distinguish trends, seasons, and residuals separately but also further determines how these three components should complement and enhance each other in the current prediction task.
This mechanism enhances the capability of STLCIAM2DNet for component collaborative modeling, avoiding the shortcomings of traditional decomposition models where "components are modeled independently at different stages with simple fusion."
Innovation Point 4: Adaptive Fusion Mechanism for Components
Under different prediction scenarios, the importance of trend components, seasonal components, and residual components for oil temperature prediction varies. For short-term oil temperature predictions, residual disturbances and recent changes may be more critical; for medium to long-term oil temperature predictions, trend components and seasonal components usually have stronger explanatory power; in scenarios with clear load cycles, the contribution of the seasonal component may be more pronounced.
STLCIAM2DNet introduces an adaptive fusion mechanism based on different samples, automatically learning the weighted fusion of trend, seasonal, and residual components. The model no longer manually sets the importance of each component but dynamically assigns weights through data-driven approaches.
This design enhances the model's adaptability to different operational states and different prediction tasks, enabling stable prediction capabilities under various scenarios such as steady operation, cycle fluctuations, and sudden disturbances.
Innovation Point 5: Two-dimensional Time Series Modeling for Multi-variable Coupling
Transformer oil temperature prediction is not a single variable prediction problem. Oil temperature changes are often related to multiple operating variables, such as high-voltage side load, low-voltage side load, medium-voltage side load, and other electrical quantities and environmental factors.
STLCIAM2DNet treats multivariate time series as a two-dimensional structure, where one dimension represents time and the other variable. Two-dimensional convolution can capture:  
1. The historical dependencies on the time dimension, such as the trend of oil temperature changes over several hours in the past;  
2. Interaction relationships between variables, such as the synergistic change between load variables and oil temperature.  
This two-dimensional modeling approach is more suitable for the prediction task of multivariate transformer oil temperature than traditional one-dimensional time convolution, which can fully utilize the relevant information between different input variables.  
2 Data Set Introduction  
2.1 Import Data  
2.2 Data STL Decomposition  
2.3 Preparation and Preprocessing of the Data Set  
Details are provided in the document provided!  
3 High-Innovative Prediction Model Based on STLCIAM2DNet  
3.1 Define the STLCIAM2DNet prediction network model  
3.2 Set parameters and train the model  
50 epochs, MSE is 0.005589, the prediction effect of STLCIAM2DNet is significant. This model decouples the trend, cycle, and disturbance features in the oil temperature sequence through STL decomposition, extracts the joint representation of time and variable through multi-scale two-dimensional convolution, enhances prediction ability through component interaction and adaptive fusion mechanism, thereby achieving more accurate, stable, and interpretable transformer oil temperature prediction.  
4 Results Visualization and Model Evaluation  
4.1 Prediction results visualization  
4.2 Model evaluation  
STLCIAM2DNet is not only applicable to transformer oil temperature prediction but also can be extended to predict power of photovoltaic, power load, equipment status, wind power power, etc., in multivariate time series prediction tasks.  
Click to download: Complete data, Python code  
Previously interesting content:  
Informer prediction model collection: Added feature importance analysis!
Fast Fourier Transform Brute Force Technique! High-Innovation Time Series Classification Model Based on the Fusion of Time-Frequency Features  
!STL-VMD Secondary Decomposition + Informer-LSTM Parallel Prediction Model
Continued updates! Collection of attention mechanisms + Lithium Battery Remaining Life Prediction  
Directly write in one area! CEEMDAN Decomposition + Informer-LSTM +XGBoost Combination Prediction Model  
Comparative Prediction Models Based on Informer-SENet for Photovoltaic Power Generation Power Prediction! Six sets of comparative prediction models, directly written for graduation thesis or minor thesis!  
!High-Innovation Model Based on ICEEMDAN+SHAP Explanability Analysis for Lithium Battery Remaining Life Prediction!
!Prediction Model Based on Informer-BiGRUGATT-CrossAttention  
High Innovation! Based on ICEEMDAN+MSCNN-BiGRU-Attention Parallel Prediction Model  
Air Quality PM2.5 Concentration High Innovation Prediction Model! Based on VMD Rolling Decomposition + Attention Mechanism's Parallel Prediction Model  
End  
\t2. Friends who like can click to follow, give the article a "like", share on WeChat Moments or discuss groups

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
