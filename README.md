# manifold-valued tensor approximation
-----

Curvature corrected tangent space-based approximation of manifold-valued data

        [1] W. Diepeveen, J. Chew, D. Needell.  
        Curvature corrected tangent space-based approximation of manifold-valued data
        arXiv preprint arXiv:xxx.xxxx. 2023 mmm dd.

Setup
-----

The recommended (and tested) setup is based on MacOS 12.3 running Julia 1.7.2. Clone the repository and activate the environment

    # Create conda environment
    conda create --name esl1 python=3.6
    conda activate esl1

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Cryo-EM.git
    cd "Cryo-EM"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

The following jupyter notebooks have been used to produce the results in [1]. 
The tables and plots are directly generated after running the notebook. 

* 6.1. Asymptotic behaviour (Tab. 1 to 6):

        experiments/testing_asymptotics/experiment.ipynb

* 6.2. Joint 3D map reconstruction and rotation estimation (Fig. 4, 5 and 6):

        experiments/testing_joint_refinement/experiment.ipynb

