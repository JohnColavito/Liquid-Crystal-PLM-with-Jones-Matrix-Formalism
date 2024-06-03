# Liquid Crystal Polarized Light Microscopy with the Jones Matrix Formalism
This repository contains a Python notebook for simulating light propagation through liquid crystal (LC) samples using the Jones Matrix Formalism. The simulation can create cross-polarized images of LC samples based on their director fields.

Prerequisites:
Python 
Pandas
NumPy
Matplotlib

For generating director field data: COMSOL

Use COMSOL to generate a text file containing the director field information of the LC sample of the form shown below. Each column should be seperated from the next by at least 4 spaces.
X    Y    Z    directorx @ t=0.001    directory @ t=0.001    directorz @ t=0.001

The code in this repository simulates light propagation through liquid crystal (LC) samples using the Jones Matrix Formalism. It can generate cross-polarized images based on the director field data of the LC sample. Additionally, it can create visualizations and videos showing how polarized light interacts with the LC sample under various conditions, aiding in the analysis of optical properties and defect structures in the LC material. An example Jupyter notebook (example_notebook.ipynb) is included in the repository to demonstrate the full workflow, from loading the director field data to generating and visualizing the cross-polarized images.
