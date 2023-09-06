# cluster_intrepretability
In this repository we find the code developed for the master's thesis "Interpreting Bayesian Network-based Clustering".
Code for solving the clustering problem with Bayesian Networks (BN) for categorical data is founded. discrete_structure.py contains all the functions needed for such purpose, whereas customers.py is an applied example on how to use these functions for obtaining the desired BN for the customers dataset. This dataset can be found in the dataset folder, we also find in the network folder the learnt BN best_network_customers_2.pickle .

Code for analyzing the respective BNs obtained is implemented in discrete_analysis_hellinger.py and discrete_representation.py . In these files we find the functions implementing the methodology developed in the thesis for cluster characterization in order to solve cluster labeling. Finally, customers_analysis.py contains an example on how to apply these functions for the network obtained in customers.py .


Moreover, continuous_structure.py and continuous_analysis.py are files where the proposed methodology and structure learning are adapted for a particular continuous case presented in the thesis with dataset example (dataset folder). This can be applied to others datasets but for problems with external libraries it may not work. 


Finally, gbf_example.py contains code for a particular example of the thesis and pybnesianCPT_to_df.py and radar_chart_discrete.py contain support functions needed for the implementations mentioned before.
