# Liquid Crystal Polarized Light Microscopy with the Jones Matrix Formalism
This repository contains a Python notebook for simulating light propagation through liquid crystal (LC) samples using the Jones Matrix Formalism. The simulation can create cross-polarized images of LC samples based on their director fields.

Prerequisites:
Python 
Pandas
NumPy
Matplotlib

For generating director field data: COMSOL

Use COMSOL to generate a text file containing the director field information of the LC sample. The names of the columns are shown below. Each element in a row should be seperated from the next by at least 4 spaces. &nbsp;&nbsp;&nbsp;&nbsp;
X &nbsp;&nbsp;&nbsp;&nbsp;
Y &nbsp;&nbsp;&nbsp;&nbsp;
Z &nbsp;&nbsp;&nbsp;&nbsp;
directorx @ t=0.001 &nbsp;&nbsp;&nbsp;&nbsp;
directory @ t=0.001 &nbsp;&nbsp;&nbsp;&nbsp;
directorz @ t=0.001 &nbsp;&nbsp;&nbsp;&nbsp;

The code in this repository simulates light propagation through liquid crystal (LC) samples using the Jones Matrix Formalism. It can generate cross-polarized images based on the director field data of the LC sample. Additionally, it can create visualizations and videos showing how polarized light interacts with the LC sample under various conditions, aiding in the analysis of optical properties and defect structures in the LC material. An example python file (example_notebook.py) is included in the repository to demonstrate the full workflow, from loading the director field data to generating and visualizing the cross-polarized images. Addititionally, an essay (LC Research Paper) is included elaborating on the implemation and depicting results from the code.
