# AI-For-Alzheimers

A python script intended to train a CNN for detecting Alzheimers early on.

# Basic Idea

1. Take a public dataset of NIFTI (Neuroimaging Informatics Technology
   Initiative) files which represent 3d and sometimes 4d (3d + temporal) brain
   scans tagged with labels indicating whether that person has been diagnosed
   with Alzheimers later in life.
2. Filter out small and poor quality images using pandas and NiBabel
3. Apply various statistical models to resulting images using NumPy
4. Feed the data and tags into the model to see the results
5. Change the dataset as needed

# Requirements

- Python
- PyTorch
- NiBabel
- Pandas
- NumPy
