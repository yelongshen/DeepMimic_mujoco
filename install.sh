#!/bin/bash

# Installation script for DeepMimic with modern MuJoCo
# This script installs all necessary dependencies

set -e  # Exit on error

echo "================================================"
echo "DeepMimic MuJoCo Installation Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "WARNING: You are not in a virtual environment!"
    echo "It's recommended to create one first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing core dependencies..."
echo ""

# Install MuJoCo
echo "Installing modern MuJoCo package..."
python3 -m pip install --upgrade mujoco

# Install GLFW for visualization
echo "Installing GLFW for visualization..."
python3 -m pip install --upgrade glfw

# Install other Python dependencies
echo "Installing other Python dependencies..."
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade gym
python3 -m pip install --upgrade pyquaternion
python3 -m pip install --upgrade joblib

# TensorFlow (optional - comment out if not needed)
echo "Installing TensorFlow..."
python3 -m pip install --upgrade tensorflow

# Check if we need to install MPI
echo ""
echo "Checking for MPI installation..."
if command -v mpirun &> /dev/null; then
    echo "MPI found, installing mpi4py..."
    python3 -m pip install --upgrade mpi4py
else
    echo "WARNING: MPI not found!"
    echo "MPI is required for parallel training."
    echo "To install on Ubuntu/Debian:"
    echo "  sudo apt-get install openmpi-bin openmpi-common openssh-client libopenmpi-dev"
    echo "Then run: pip install mpi4py"
    echo ""
fi

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "To test the installation, run:"
echo "  cd src"
echo "  python3 dp_env_v3.py"
echo ""
echo "For more information, see MIGRATION_GUIDE.md"
echo ""
