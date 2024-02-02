# ChemPredictor
![alt text](https://github.com/CubeStar1/ChemPredictorv2/blob/master/utilities/assets/logo/ChemPredictor_flat_logo_wide.jpg?raw=true)

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Key Features](#key-features)
4. [Usage](#usage)
    1. [Installation and Setup](#installation-and-setup)
    2. [How to Use](#how-to-use)
5. [Dependencies](#dependencies)
7. [File Structure](#file-structure)
8. [Hosted Version](#hosted-version)

## Overview 

ChemPredictor aims to analyze Molecular Properties of known compounds and construct an advanced Artificial Neural Network (ANN) model capable of accurately predicting these properties for unknown compounds. This project seamlessly integrates the principles of chemistry, mathematics, and Python programming to develop and deploy an ANN model for Molecular Property Prediction (MPP).

### Objectives 

1. **Molecular Property Analysis:** Explore the Molecular Properties of known compounds.
2. **ANN Model Construction:** Build a robust Artificial Neural Network model for accurate prediction of molecular properties.
3. **Interdisciplinary Approach:** Integrate concepts from chemistry, mathematics, and programming to enhance the model's effectiveness.
4. **User-Friendly Interface:** Develop an interactive [WebUI using Streamlit](https://chempredictor.streamlit.app/) for seamless and user-friendly predictions.
5. **Targeted Properties:** Focus on predicting a set of 9 molecular properties crucial for comprehensive chemical analysis:

   **Table 1** - Predicted properties of the QM9 dataset
   
   | No. | Property | Unit      | Description                             |
   |-----|----------|-----------|-----------------------------------------|
   | 1   | μ        | D         | Dipole moment                           |
   | 2   | α        | a³        | Isotropic polarizability                |
   | 3   | homo     | Ha        | Energy of HOMO                          |
   | 4   | lumo     | Ha        | Energy of LUMO                          |
   | 5   | gap      | Ha        | Gap (lumo − homo)                       |
   | 6   | U        | Ha        | Internal energy at 298.15 K             |
   | 7   | H        | Ha        | Enthalpy at 298.15 K                    |
   | 8   | G        | Ha        | Free energy at 298.15 K                 |
   | 9   | Cv       | cal/mol K | Heat capacity at 298.15 K               |

    - **Dipole moment (µ):** Measurement of polarity of a molecule.
    - **Electronic polarizability (α):** Tendency of non-polar molecules to shift their electron clouds relative to their nuclei.
    - **Energy of HOMO:** Energy of the highest occupied Molecular Orbital.
    - **Energy of LUMO:** Energy of the lowest unoccupied Molecular Orbital.
    - **Band Gap Energy:** Energy of LUMO – Energy of HOMO.
    - **Internal energy of atomization (U):** Energy required to break a molecule into separate atoms.
    - **Enthalpy of atomization (H):** Amount of enthalpy change when a compound's bonds are broken, and the component atoms are separated into single atoms.
    - **Free energy of atomization (G):** Extra energy needed to break up a molecule into separate atoms.
    - **Heat capacity (Cv):** Amount of heat required to increase the temperature of the molecule by one degree.

### Key Features 

- **Multiple Input Options:** ChemPredictor supports various input methods for user convenience.

  - **SMILES Input:** Predict properties by entering the SMILES string of the compound directly.
  - **Common Name Input:** Input the common name of the compound (e.g., \"Aspirin\") to predict its properties.
  - **CSV File Upload:** Upload a CSV file containing SMILES strings to predict properties for multiple molecules.
  - **Molecule Drawing:** Utilize an interactive drawing board to draw the molecule for property prediction.

- **Similar Molecules:** Explore similar molecules based on the predicted properties, providing valuable insights into chemical similarity.

- **3D Molecular Visualization:** Visualize molecular structures in three dimensions, enhancing the understanding of the compound's spatial arrangement."




## Usage


### Installation and Setup

To run the ChemPredictor WebUI, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CubeStar1/ChemPredictorv2.git
   cd ChemPredictorv2
2. **Create a virtual environment:**

   ```bash
   python -m venv venv
3. **Activate the virtual environment:**

   - On Windows:   
      ```bash
     .\venv\Scripts\activate
   - On Unix or MacOS:
      ```bash
     source venv/bin/activate
4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   
### How to Use
1. **Run the Streamlit app:**

   ```bash
    streamlit run app.py
2. **Open the WebUI in your browser:** 

   - Open your web browser and navigate to http://localhost:8501 to use the ChemPredictor WebUI.

3. **Predict Molecular Properties:**

   Choose one of the following ways to predict molecular properties:
    - Enter Common Name : Input the common name of the compound (e.g., \"Aspirin\") to predict its properties.

   - Enter SMILES String:

     - Enter the SMILES string of the compound in the provided input field.
   - Upload SMILES CSV File:

      - Upload a CSV file containing a set of SMILES strings for bulk prediction.
   - Draw Molecule:

     - Utilize the interactive drawing board to draw the molecule for prediction.
4. **Click on the Predict Button:**
   - Once the input is provided (SMILES string, CSV file, or drawn molecule), click on the "Predict" button.
5. **View Predicted Values:**
   - The predicted values will be displayed on the right side of the screen.

### Dependencies

#### Dataset
- **QM9 Dataset:** ChemPredictor utilizes the QM9 dataset, a benchmark in quantum chemistry, for training and testing the model. The dataset includes thermochemical properties, electronic properties, and geometries of small organic molecules.

#### Data Analysis and Exploration
- **Libraries Used:** NumPy, Pandas, Matplotlib
- **Overview:** Statistical summaries are generated using NumPy and Pandas to gain insights into data distribution, missing values, and outliers. Matplotlib is employed for data visualization to uncover patterns and relationships.

#### Data Preparation
- **Libraries Used:** RDKit, Mordred Molecular Descriptors
- **Overview:** Feature engineering is carried out using RDKit to create 2D representations of molecules from SMILES input. RDKit and Mordred Molecular Descriptors are used for preprocessing, generating descriptors for each input molecule. The dataset is then split into training, validation, and test sets.

#### Machine Learning
- **Libraries Used:** TensorFlow with Keras, Scikit-learn
- **Overview:**  An Artificial Neural Network (ANN) is developed using TensorFlow with Keras. The model is trained on the training data, and its performance is evaluated using Scikit-learn metrics.

#### Model Deployment and Prediction
- **Libraries Used:** Streamlit
- **Overview:** A user-friendly web interface is created using the Streamlit framework for model deployment. User input is accepted via the WebUI, preprocessed, and predictions are made using the deployed ANN model. 

## File Structure
```
project-root/
├── app.py                       # Streamlit WebUI
│
├── scripts/
│   ├── image_handling.py        # Image handling functions
│   ├── predict_property         # Property prediction functions
│   ├── project_overview_page.py # Project overview page
│   ├── utils.py                 # Project utility functions
│
├── utilities/
│   ├── assets/                 # Images
│   ├── dataset/                # QM9 dataset
│   ├── descriptors/            # Mordred descriptor files
│   ├── models/                 # Trained model files
│   ├── scalers/                # Pickled MinMaxScaler file
│
├── images/                     # Image storage 
│   
├── requirements.txt
├── packages.txt
│
└── README.md
```

## Hosted Version
For a quick demo, you can also access the hosted version of ChemPredictor at https://chempredictor.streamlit.app/