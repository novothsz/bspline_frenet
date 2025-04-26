#!/usr/bin/env bash

# # Create new environment with specified Python version
# conda create -n bspline_frenet python=3.12 -c conda-forge -y

# # Activate the environment
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate bspline_frenet

# # Install core numerical packages
# conda install -c conda-forge numpy=1.26.* scipy=1.13.* matplotlib=3.4.3 autograd=1.5.* casadi=3.6.* -y

# # (Optional) Install additional dependencies if you uncomment them
# # conda install -c conda-forge scikit-learn numba pyqt imageio ffmpeg -y

#!/usr/bin/env bash

# Create environment with Python 3.10 (compatible with matplotlib 3.4.3)
conda create -n bspline_frenet python=3.10 -c conda-forge -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bspline_frenet

# Install core packages with pinned versions
conda install -c conda-forge \
    numpy=1.26.* \
    scipy=1.13.* \
    matplotlib=3.4.3 \
    autograd=1.5.* \
    casadi=3.6.* \
    pyyaml \
    -y

conda install -c gurobi gurobi -y