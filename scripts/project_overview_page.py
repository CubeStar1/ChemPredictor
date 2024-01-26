import streamlit as st
def project_overview():
    st.title("Project Overview")
    st.markdown("---")
    # content, image = st.columns([2,1])
    # with content:
    st.markdown(""" 
    ##  Molecular Property Prediction using Artificial Neural Networks
    This project focuses on leveraging ANN's for the prediction of molecular properties and the application of ANN's 
    in the field of computational chemistry
    and holds the potential to revolutionise the understanding and prediction of key molecular characteristics 
    """)
    st.image('utilities/assets/overview_images/mpp_image.png')
    st.markdown("""
    ## Objective
    
    - The objective of this project is to analyze the molecular properties of known compounds and build an Artificial 
    Neural Network (ANN) model that can accurately predict these properties for unknown compounds. 
    - The project integrates concepts from chemistry, mathematics, and Python programming to develop and deploy an ANN model for 
    Molecular Property Prediction (MPP). 
    - An interactive WebUI will be created using Streamlit to facilitate user-friendly predictions. 
    - The final goal is to predict a set of 9 targeted molecular properties.
        
    ## Implementation
    
    To develop a computational model to predict properties, the properties need to be described in ways that can be 
    tied to a chemical or physical property. There are many ways to represent organic molecules. To make a 
    reasonable prediction for any set of molecules, the physical or chemical data must be related to the molecule 
    through a series of descriptors. These descriptors can be structural, relating data about the relative position 
    of atoms and types, or calculated data such as electron density using quantum chemical methods.
    
    SMILES (Simplified Molecular Input Line Entry System) is a language for molecular representation with its own 
    semantics and grammar. It is a string-based representation that encodes the structures into compact character 
    sequences obtained from 2D molecular graphs. In SMILES notation, atoms are represented by their elemental 
    symbols and bonds are indicated by dashes and colons. For example, the SMILES representation of the benzene 
    molecule is ‘c1ccccc1’, where ‘c’ represents the atom of carbon and the number 1 the beginning and end of a 
    cyclic structure.

    The SMILES strings from the dataset will be converted into 2D molecular representations using an open-source 
    library called RDKit. These 2D molecules are then passed as inputs into the Mordred Molecular Descriptor, 
    which generates 1000+ molecular descriptions for each of the input SMILES strings.
    """)
    st.image('utilities/assets/overview_images/mol_descriptors.png')


    st.markdown("""

    ## The Chemistry Behind MPP
    
    The model will be trained to predict the following properties:
    - Dipole moment (µ)
    - Electronic polarizability (α)
    - Internal energy of atomization (U)
    - Enthalpy of atomization (H), 
    - Free energy of atomization (G)
    - Energy of HOMO
    - Energy of LUMO
    - Band Gap Energy
    - Heat capacity (Cv)

    The QM9 dataset, which holds information on the energetic, electronic, and thermodynamic properties of 134,000 
    molecules composed of C, H, O, N, and P atoms, will be used. The original dataset contained about 130,000 
    molecules, approximately 35% of which contained 17-19 atoms.
    """)

    st.image('utilities/assets/overview_images/qm9_graph.png')

    st.markdown('''
        

    ## The Mathematics Behind ANN's

    The project involves the use of Artificial Neural Networks (ANNs) to predict molecular properties. 
    The mathematics behind ANNs involve linear algebra for data representation, calculus for minimizing the cost 
    function through algorithms like gradient descent and backpropagation, and statistics for optimizing the 
    weights of the neural network.
    The architecture of an ANN includes neurons, weights, an input layer, one or more hidden layers, and an output 
    layer. The activation function is a crucial component that introduces non-linearity to the model, enabling it to 
    solve complex problems. Various types of activation functions are used, including Sigmoid, Tanh, ReLU, Leaky 
    ReLU, Swish, and ELU.
    
    Lets see how forward propagation works in an ANN, the weighted sum is computed from the input layer and it is 
    run through an activation function that presents non linearity to the model, this process is carried out for 
    every neuron in every layer and we then receive the output in the final layer.
    
    To understand how a neural network actually learns we will have to see what backward propagation is, we compare 
    the initial output of a neural network to the actual value and then compute the cost function that is the error 
    of the algorithm using the sum of squares method, then we implement the gradient descent algorithm using the 
    chain rule of calculus and we minimise the gradient of the function
    till we get to zero which means that the output of the network is accurate. 
    ''')
    st.image('utilities/assets/overview_images/ann_example.png')

    st.markdown("""

    ## Implementation Using Python
    The implementation process involves several steps. First, data analysis and exploration are performed using 
    libraries like NumPy, Pandas, and Matplotlib. Statistical summaries are generated to understand data 
    distribution, missing values, and outliers. Data visualization techniques are used to visualize patterns and 
    relationships in the data.

    In the data preparation stage, feature engineering is performed using the RDKit library to create 2D 
    representations of molecules from SMILES input. Preprocessing is applied using RDKit and Mordred Molecular 
    Descriptors to generate descriptors for each input molecule. The dataset is then split into training, 
    validation, and test sets.

    In the machine learning stage, various algorithms are experimented with for model selection. An ANN is developed 
    using TensorFlow with Keras as a framework. The model is trained on the training data and its performance is 
    evaluated using Scikit-learn metrics.

    Finally, in the model deployment and prediction stage, a simple web interface is created using the Streamlit 
    framework. User input is accepted via the WebUI, preprocessed, and predictions are made using the deployed model. 
    This comprehensive approach integrates various disciplines and techniques to achieve accurate molecular property prediction.
    
    The dataset is loaded using libraries like NumPy, Pandas, and Matplotlib. An ANN with 1000 input layers, 
    64 hidden layers, and 1 output layer is developed, and nine different ANNs are trained using TensorFlow with 
    Keras as a framework. The model’s performance is evaluated using Scikit-learn metrics such as RMSE, accuracy, 
    precision, recall, etc. The trained model is saved for future use.

    
    A simple web interface is developed using the Streamlit framework for model deployment and prediction. The WebUI 
    accepts user input in the form of SMILES, either as direct input or file upload, and even allows users to draw 
    molecules as input. It then preprocesses the input and makes predictions using the deployed model. The WebUI 
    also has features to show similar molecules, 3D molecules, and predicted properties.
    
    """)
    st.image('utilities/assets/overview_images/model_architecture.png')

    st.markdown("""

    ## Applications and Relevance to Society
    
    Molecular property prediction has a wide range of applications across various scientific and industrial domains
    especially in drug discovery, where it can help in Bioactivity Prediction which is Predicting the biological 
    activity of molecules that helps identify potential drug candidates and
    Toxicity Prediction that helps in Assessing the potential toxicity of drug candidates early in the development 
    process.

    It also has applications in material science where its helpful in Material Design where it helps in Predicting 
    material properties, such as strength, conductivity, and thermal stability, to design new materials with 
    specific characteristics and Catalyst Design which is Predicting the catalytic activity of molecules for use in 
    chemical processes.

    It has wide ranging applications in the fields of biotechnology and polymer chemistry where its helpful in 
    enzyme engineering, predicting protein-ligand interactions and polymer design.
    
    ## Conclusion
     This project aims to address the limitations of traditional methods for predicting molecular properties by 
     leveraging the power of ANNs and developing an interactive WebUI. By using SMILES, graphs, or physical organic 
     descriptors for molecular representation, a good platform for featurization is provided, serving as the input 
     for various ML algorithms. This approach has the potential to revolutionize various industries, including drug discovery, 
     materials science, and environmental studies.
     """)

