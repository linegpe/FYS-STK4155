### DATA ANALYSIS WITH LINEAR REGRESSION METHODS WITH APPLICATION OF RESAMPLING TECHNIQUES
#### APPLIED DATA ANALYSIS AND MACHINE LEARNING, FYS-STK4155
This folder contians all codes written for Project 1, made in cooperation with Maria Markova

The folder contains the programs used to collect the results for the project:
  1. main.py is used to collect results for Franke's function
  2. main_geo.py is used to collect results for terrain data
  3.statistical_functions.py contain all nesessary statistical functions (e.g. R2, MSE)
  4. resampling_methods.py contains bootstrap function, cross-validation function and no-resampling option
  5. regression_methods.py contains OLS, Ridge, LASSO(SKL) and other SKL functions
  6. print_and_plot.py is used for plotting all results
  7. data_processing is used to set and rescale data
  
 File folder: all files used for plotting
 Figures: all plotted figures
 
 Example code:
  1. OLS_CV_Complexity - performs the complexity study with OLS and 5-fold cross-validation
  2. Ridge_bootstrap_optimal_lambda - reproduces the optimal lambda value for ridge+bootstrap
  3. LASSO_bootstrap_optimal_lambda - reproduces the optimal lambda value for ridge+bootstrap
  4. Terrain_complexity_plus_grid_search - complexity study for terrain data with ridge and 8-fold cross-validation, 
     search for the optimal lambda for polynomial degree 10 and grid search with ridge
