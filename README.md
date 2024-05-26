# ChemPredictor
![alt text](https://github.com/CubeStar1/ChemPredictorv2/blob/master/utilities/assets/logo/ChemPredictor_flat_logo_wide-modified.png?raw=true)

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
9. [How it Works](#how-it-works)
## Overview 

- ChemPredictor is a Streamlit Web Interface that uses an artificial neural network (ANN) trained using TensorFlow for quantitative structure-property relationship (QSPR) analysis of molecules to predict 9 thermodynamic properties. 
- Users can either enter the common name of the compound or draw the molecule using an interactive sketch tool and obtain predicted properties in real time.
- The project aims to provide a user-friendly interface for chemists, researchers, and students to analyze and predict the properties of organic molecules. 
- The integration of Google's AI language model, Gemini Pro, allows users to obtain additional information about the predicted properties or the compound itself. (API Key required)
### Objectives 

- **Predict Molecular Properties:** Utilize an ANN model to predict 9 thermodynamic properties of organic molecules based on their molecular structure.
- **User-Friendly Interface:** Develop a Streamlit WebUI that allows users to input the common name, SMILES string, or draw the molecule for property prediction.
- **Gemini LLM Integration:** Integrate Google's AI language model, Gemini Pro, to provide additional information about the predicted properties or the compound itself. (API Key required)
![alt text](https://github.com/CubeStar1/ChemPredictorv2/blob/master/utilities/assets/overview_images/implementation_flowchart.jpg?raw=true)
![alt text](https:///github.com/CubeStar1/ChemPredictorv2/blob/master/utilities/assets/overview_images/model_architecture.png?raw=true)
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
- **Gemini LLM Integration:** 
- **Multiple Input Options:** ChemPredictor supports various input methods for user convenience.

  - **SMILES Input:** Predict properties by entering the SMILES string of the compound directly.
  - **Common Name Input:** Input the common name of the compound (e.g., \"Aspirin\") to predict its properties.
  - **CSV File Upload:** Upload a CSV file containing SMILES strings to predict properties for multiple molecules.
  - **Molecule Drawing:** Use an interactive drawing board to draw the molecule for property prediction.

- **Similar Molecules:** Shows similar molecules based on the predicted properties, providing insights into chemical similarity.

- **3D Molecular Visualization:** Shows molecular structures in three dimensions, providing an insight into the compound's spatial arrangement."





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
![alt text](https://github.com/CubeStar1/ChemPredictorv2/blob/master/utilities/assets/overview_images/web_interface.png?raw=true)

### Dependencies

#### Dataset
- **QM9 Dataset:** The QM9 dataset is used for training the ANN model. It contains 133,885 stable small organic molecules with up to nine heavy atoms (C, N, O, F). The dataset includes 9 thermodynamic properties for each molecule.

#### Data Analysis and Exploration
- **Libraries Used:** NumPy, Pandas, Matplotlib
- **Overview:** Data analysis and exploration are performed using NumPy, Pandas, and Matplotlib to understand the dataset's structure and properties.

#### Data Preparation
- **Libraries Used:** RDKit, Mordred Molecular Descriptors
- **Overview:** RDKit is used to convert the SMILES strings into molecular structures, and Mordred is used to calculate 1826 molecular descriptors for each molecule.
#### Machine Learning
- **Libraries Used:** TensorFlow with Keras, Scikit-learn
- **Overview:**  An ANN model is trained using TensorFlow with Keras to predict the 9 thermodynamic properties of organic molecules. The model is trained on the QM9 dataset and evaluated using Scikit-learn.
#### Model Deployment and Prediction
- **Libraries Used:** Streamlit
- **Overview:** Streamlit is used to make a user-friendly Web Interface. User input is accepted via the WebUI, preprocessed, and predictions are made using the deployed ANN model. 
- **Gemini LLM Integration:** Google's AI language model, Gemini Pro, is integrated to provide additional information about the predicted properties or the compound itself. (API Key required)

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

## How it Works

![alt text](https://github.com/CubeStar1/ChemPredictorv2/blob/master/utilities/assets/overview_images/web_interface_working.png?raw=true)