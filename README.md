# manifold-valued tensor approximation

This repository contains the code for the main algorithms from

        [1] W. Diepeveen, J. Chew, D. Needell.  
        Curvature corrected tangent space-based approximation of manifold-valued data
        arXiv preprint arXiv:2306.00507. 2023 June 1.

Setup
-----

The recommended (and tested) setup is based on MacOS 12.3 running Julia 1.7.2. Clone the repository and activate the environment

    julia> ]
    (v1.7) pkg> activate .
    (env)  pkg> instantiate


Reproducing the experiments in [1]
----------------------------------

The following jupyter notebooks have been used to produce the results in [1]. 
The tables and plots are directly generated after running the notebook. 

* 6.1 Synthetic low-rank 1D signals.
   * 6.1.1 Spherical data. (Fig. 2 and 3, and Tab. 1 and 2):

           experiments/1D/S6/CCLRA_artifical_1D.ipynb
   * 6.1.2 Symmetric positive definite matrix data. (Fig. 5 and Tab. 3):

           experiments/1D/P3/CCLRA_artifical_1D.ipynb

* 6.2. Real 2D DT-MRI data. (Fig. 7 and 8, and Tab. 4 and 5):

        experiments/2D/P3/DTI_camino_2D.ipynb

