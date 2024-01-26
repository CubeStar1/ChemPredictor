# Libraries
# ---------------------------------------------
# General
import os
import glob
from typing import Any, Dict
# Data manipulation
import pandas as pd   # To read the dataset
import joblib # To load the scaler
# Streamlit - To build the web application
import streamlit as st
import streamlit_antd_components as sac
# Ketcher - To draw molecules
from streamlit_ketcher import st_ketcher
# 3Dmol - To display molecules
# from utils import display_3D_molecule
# from stmol import showmol
# import py3Dmol
# from about_page import project_overview
# Utilities
from scripts import project_overview_page
from scripts import utils
from scripts.utils import get_smiles_from_name
from scripts.utils import display_molecule_in_dataframe_as_html
from scripts.predict_property import generate_prediction_dataframe
# RDKit - To handle molecules
from rdkit import Chem# To extract information of the molecules
# Tensorflow - To load the neural network
from tensorflow import keras # To build the neural network
# Sklearn - To load the scaler
from sklearn.preprocessing import MinMaxScaler
# ---------------------------------------------


st.set_page_config(page_title="Molecular Properties Prediction App", layout='wide', page_icon=":bar_chart:")
# Monkey patching the MinMaxScaler to avoid clipping
original_minmax_setstate = MinMaxScaler.__setstate__
def __monkey_patch_minmax_setstate__(self, state: Dict[str, Any]) -> None:
    state.setdefault("clip", False)
    original_minmax_setstate(self, state)
MinMaxScaler.__setstate__ = __monkey_patch_minmax_setstate__

@st.cache_data
def load_descriptor_columns():
    desc_df = pd.read_csv('utilities/descriptors/descriptor_columns_full1.csv')
    desc_df_columns = desc_df['descriptor'].tolist()
    return desc_df_columns

# Loading the descriptor columns
# desc_df = pd.read_csv('utilities/descriptors/descriptor_columns_full1.csv')
desc_df_columns = load_descriptor_columns()


def load_models():
    # Loading the scaler and the models
    scaler = joblib.load('utilities/scalers/ann_scaler_cv_full (1).joblib')
    model_cv = keras.models.load_model('utilities/models/ann_cv_model_full.h5')
    model_G = keras.models.load_model('utilities/models/ann_G_model_full.h5')
    model_mu = keras.models.load_model('utilities/models/ann_mu_model_full.h5')
    model_homo = keras.models.load_model('utilities/models/ann_homo_model_full.h5')
    model_lumo = keras.models.load_model('utilities/models/ann_lumo_model_full.h5')
    model_alpha = keras.models.load_model('utilities/models/ann_alfa_model_full.h5')
    return scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha


scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha = load_models()

# Defining which property to predict
properties = ['HOMO', 'LUMO', 'Band Gap', 'Polarizability', 'Dipole moment', 'U', 'H', 'G','Cv']

# Defining the layout of the web application
# TITLE
# st.set_page_config(page_title="Molecular Properties Prediction App", layout='wide', page_icon=":bar_chart:")
# st.title('Molecular Properties Prediction App')
# st.markdown("""---""")

# SIDEBAR

with st.sidebar:
    with st.container(border=True):
        st.markdown('<h2 style="text-align: center;font-size: 1.5em;">Molecular Property Predictor</h2>', unsafe_allow_html=True)
        st.image('utilities/assets/logo/ChemPredictorLogo.png')

with st.sidebar:
    #logo_smile = "c1ccccc1"
    #display_3D_molecule(logo_smile, width=200, height=200)

    with st.expander("Property Selection", expanded=False):
        property_selection = sac.chip(items=[
            sac.ChipItem(label='HOMO'),
            sac.ChipItem(label='LUMO'),
            sac.ChipItem(label='Band Gap'),
            sac.ChipItem(label='Cv'),
            sac.ChipItem(label='U'),
            sac.ChipItem(label='H'),
            sac.ChipItem(label='G'),
            sac.ChipItem(label='Polarizability'),
            sac.ChipItem(label='Dipole Moment'),
        ], label='### Select Properties',
            index=[3, 6, 8],
            format_func='title', align='center',
            multiple=True, radius='md',
            variant='filled',
    )
    #st.markdown("""---""")
    # Model Selection
    with st.expander('Model Selection', expanded=False):

        # st.markdown("## Select a model")
        #model_selection = st.selectbox("",('Artificial Neural Network', 'Random Forest', 'Support Vector Machine'))
        model_selection = sac.chip(items=[
            sac.ChipItem(label='Artificial Neural Network')],
            # sac.ChipItem(label='Random Forest'),
            # sac.ChipItem(label='Support Vector Machine')],
                        label='### Select Model',
                        index=[0],
                        format_func='title',
                        align='start',
                        multiple=False,
                        radius='md',
                        variant='filled')

    with st.expander("Input Selection", expanded=True):
        # st.markdown("## Select input type")
        input_selection = sac.chip(items=[
            sac.ChipItem(label='Common Name'),
            sac.ChipItem(label='SMILES input'),
            sac.ChipItem(label='Upload SMILES as file input'),
            sac.ChipItem(label='Draw molecule')],
                        label='### Select Input Type',
                        index=[0], format_func='title',
                        align='center',
                        multiple=False,
                        radius='md',
                        variant='filled')
    #st.markdown("""---""")
    # Input Selection
    #st.markdown("## Select input type")
    #input_selection = st.selectbox("",('SMILES input', 'Upload SMILES as file input', 'Draw molecule'))
# 1. SMILES input
if input_selection == 'SMILES input':
    smiles_input = st.sidebar.text_input('Please input SMILE strings of the molecules in the box below:',
                                          "")
    prediction = st.sidebar.button('Predict property of molecule(s)', use_container_width=True, type='primary')
# 2. Upload SMILES as file input
if input_selection == 'Upload SMILES as file input':
    many_SMILES = st.sidebar.file_uploader('Upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in \'SMILES\' column:')
    prediction = st.sidebar.button(f'Predict property of molecules', type='primary', use_container_width=True)

if input_selection == 'Common Name':
    common_name = st.sidebar.text_input('Please input common name of the molecule in the box below:',
                                        "")
    prediction = st.sidebar.button('Predict property of molecule', use_container_width=True, type='primary')


# PREDICTION PAGE

def Prediction():


    if input_selection == 'Common Name':
        with st.expander("How to Make Predictions", expanded=True):
            st.info("1. Input common name of the molecule in the box on the left")
            st.info("2. Click the button below to get the prediction")
        if prediction:
            files = glob.glob('images/*.png')
            for f in files:
                os.remove(f)
            smiles_string = get_smiles_from_name(common_name)
            # if smiles_string is None:
            #     st.error("No smiles string found")
            #     return
            # else:
            smiles_list = smiles_string.split(",")
            df_original = pd.DataFrame (smiles_list, columns =['SMILES'])
            output_df = generate_prediction_dataframe(df_original, desc_df_columns, scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha)
            html_df = display_molecule_in_dataframe_as_html(output_df)

            # with st.expander("Show predicted properties", expanded=True):
            #     st.markdown(f'<div id="" style="overflow:scroll; height:300px; padding-left: 20px; ">{html_df}</div>',
            #                 unsafe_allow_html=True)
            molecular_weight = str(round(output_df['Molecular Weight'].tolist()[0], 2)) + ' g/mol'

            predicted_Cv = str(output_df['Predicted Cv (cal/mol.K)'].tolist()[0]) + ' cal/mol.K'
            predicted_G = str(output_df['Predicted G (Ha)'].tolist()[0]) + ' kcal/mol'
            predicted_H = str(round(output_df['Predicted G (Ha)'].tolist()[0] - 0.21, 3)) + ' kcal/mol'
            predicted_U = str(output_df['Predicted G (Ha)'].tolist()[0] + 1.25) + ' kcal/mol'
            predicted_mu = str(output_df['Predicted mu (D)'].tolist()[0]) + ' D'
            predicted_homo = str(output_df['Predicted HOMO (Ha)'].tolist()[0]) + ' eV'
            predicted_lumo = str(round(output_df['Predicted LUMO (Ha)'].tolist()[0], 4)) + ' eV'
            predicted_bandgap = str(
                output_df['Predicted LUMO (Ha)'].tolist()[0] - output_df['Predicted HOMO (Ha)'].tolist()[0]) + ' eV'
            predicted_alpha = str(output_df['Predicted alpha (a³)'].tolist()[0]) + ' Å³'

            predicted_property_names = ['Cv', 'G', 'Dipole Moment', 'U', 'H', 'Polarizability', 'Band Gap', 'HOMO',
                                        'LUMO']
            predicted_property_values = [predicted_Cv, predicted_G, predicted_mu, predicted_U, predicted_H,
                                         predicted_alpha, predicted_bandgap, predicted_homo, predicted_lumo]

            with st.container(border=True):
                    st.markdown("## Predicted Properties")
                    structure, properties = st.columns([1,2])

                    with structure:
                        with st.container(border=True):
                            st.info('Structure')
                            images = glob.glob('images/*.png')
                            st.image(images)

                        with st.container(border=True):
                            st.info('Molecular Weight')
                            st.markdown(f'<div id="" style="display: flex; justify-content: center; align-items: center; font-size: 20px; height:100px; ">{molecular_weight}</div>', unsafe_allow_html=True)

                    with properties:
                        with st.container(border=True):
                            col1, col2, col3 = st.columns(3)
                            for i, col in enumerate([col1, col2, col3]):
                                with col:
                                    for j in range(3):
                                        with st.container(border=True):
                                            st.markdown(f'<div id="" style=" height:50px; background-color: #ff4b4b; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-weight: bold">{predicted_property_names[3*i+j]}</div>',
                                                        unsafe_allow_html=True)
                                            #st.success(f'{predicted_property_names[3*i+j]}')
                                            st.markdown(
                                                f'<div id="" style="display: flex; justify-content: center; align-items: center; font-size: 20px; height:100px; ">{predicted_property_values[3 * i + j]}</div>',
                                                unsafe_allow_html=True)




    elif input_selection == 'SMILES input' :

        with st.expander("How to Make Predictions", expanded=True):
            st.info("1. Input SMILES string(s) of the molecule(s) in the box on the left")
            st.info("2. Click the button below to get the prediction")
        if prediction:
            files = glob.glob('images/*.png')
            for f in files:
                os.remove(f)
            smiles_list = smiles_input.split(",")
            df_original = pd.DataFrame (smiles_list, columns =['SMILES'])
            output_df = generate_prediction_dataframe(df_original, desc_df_columns, scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha)
            html_df = display_molecule_in_dataframe_as_html(output_df)

            # with st.expander("Show predicted properties", expanded=True):
            #     st.markdown(f'<div id="" style="overflow:scroll; height:300px; padding-left: 20px; ">{html_df}</div>',
            #                 unsafe_allow_html=True)
            molecular_weight = str(round(output_df['Molecular Weight'].tolist()[0], 2)) + ' g/mol'

            predicted_Cv = str(output_df['Predicted Cv (cal/mol.K)'].tolist()[0]) + ' cal/mol.K'
            predicted_G = str(output_df['Predicted G (Ha)'].tolist()[0]) + ' kcal/mol'
            predicted_H = str(round(output_df['Predicted G (Ha)'].tolist()[0] - 0.21, 3)) + ' kcal/mol'
            predicted_U = str(output_df['Predicted G (Ha)'].tolist()[0] + 1.25) + ' kcal/mol'
            predicted_mu = str(output_df['Predicted mu (D)'].tolist()[0]) + ' D'
            predicted_homo = str(output_df['Predicted HOMO (Ha)'].tolist()[0]) + ' eV'
            predicted_lumo = str(round(output_df['Predicted LUMO (Ha)'].tolist()[0], 4)) + ' eV'
            predicted_bandgap = str(
                output_df['Predicted LUMO (Ha)'].tolist()[0] - output_df['Predicted HOMO (Ha)'].tolist()[0]) + ' eV'
            predicted_alpha = str(output_df['Predicted alpha (a³)'].tolist()[0]) + ' Å³'

            predicted_property_names = ['Cv', 'G', 'Dipole Moment', 'U', 'H', 'Polarizability', 'Band Gap', 'HOMO',
                                        'LUMO']
            predicted_property_values = [predicted_Cv, predicted_G, predicted_mu, predicted_U, predicted_H,
                                         predicted_alpha, predicted_bandgap, predicted_homo, predicted_lumo]

            with st.container(border=True):
                    st.markdown("## Predicted Properties")
                    structure, properties = st.columns([1,2])

                    with structure:
                        with st.container(border=True):
                            st.info('Structure')
                            images = glob.glob('images/*.png')
                            st.image(images)

                        with st.container(border=True):
                            st.info('Molecular Weight')
                            st.markdown(f'<div id="" style="display: flex; justify-content: center; align-items: center; font-size: 20px; height:100px; ">{molecular_weight}</div>', unsafe_allow_html=True)

                    with properties:
                        with st.container(border=True):
                            col1, col2, col3 = st.columns(3)
                            for i, col in enumerate([col1, col2, col3]):
                                with col:
                                    for j in range(3):
                                        with st.container(border=True):
                                            st.markdown(f'<div id="" style=" height:50px; background-color: #ff4b4b; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-weight: bold">{predicted_property_names[3*i+j]}</div>',
                                                        unsafe_allow_html=True)
                                            #st.success(f'{predicted_property_names[3*i+j]}')
                                            st.markdown(
                                                f'<div id="" style="display: flex; justify-content: center; align-items: center; font-size: 20px; height:100px; ">{predicted_property_values[3 * i + j]}</div>',
                                                unsafe_allow_html=True)

            # col1, col2 = st.columns(2)
            # with st.container(border=True):
            #     with col1:
            #         with st.container(border=True):
            #             st.info('Molecular Weight')
            #             st.metric(label="Molecular Weight",
            #                       value=molecular_weight,
            #                       label_visibility='collapsed')
            #
            #     with col2:
            #         with st.container(border=True):
            #             st.info('Structure')
            #             images = glob.glob('images/*.png')
            #             st.image(images)
            #
            #
            # with st.container(border=True):
            #     col1, col2, col3 = st.columns(3)
            #     for i, col in enumerate([col1, col2, col3]):
            #         with col:
            #             for j in range(3):
            #                 with st.container(border=True):
            #                     st.markdown(
            #                         f'<div id="" style=" height:50px; background-color: #ff4b4b; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-weight: bold">{predicted_property_names[3 * i + j]}</div>',
            #                         unsafe_allow_html=True)
            #                     # st.success(f'{predicted_property_names[3*i+j]}')
            #                     st.markdown(
            #                         f'<div id="" style="display: flex; justify-content: center; align-items: center; font-size: 20px; height:100px; ">{predicted_property_values[3 * i + j]}</div>',
            #                         unsafe_allow_html=True)
                                #st.metric(label="Predicted", value=predicted_property_values[3*i+j], delta="", label_visibility='collapsed')




    elif input_selection == 'Upload SMILES as file input' and prediction:
        files = glob.glob('images/*.png')
        for f in files:
            os.remove(f)
        df = pd.read_csv(many_SMILES)
        output_df = generate_prediction_dataframe(df, desc_df_columns, scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha)
        html_df = display_molecule_in_dataframe_as_html(output_df)
        st.markdown(html_df,unsafe_allow_html=True)


    elif input_selection == 'Draw molecule':
        # famous_molecules = [
        #     ('☕', 'Caffeine'),
        #     ('🥱', 'Melatonin'),
        #     ('🚬', 'Nicotine'),
        #     ('🌨️', 'Cocaine'),
        #     ('💊', 'Aspirin'),
        #     ('🍄', 'Psilocybine'),
        #     ('💎', 'Lysergide')
        # ]
        st.session_state.molfile = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1"))
        # for mol, column in zip(famous_molecules, st.columns(len(famous_molecules))):
        #     with column:
        #         emoji, name = mol
        #
        #         if st.button(f'{emoji} {name}'):
        #             st.session_state.molfile, st.session_state.chembl_id = utils.name_to_molecule(name)

        files = glob.glob('images/*.png')
        for f in files:
            os.remove(f)
        # st.markdown("""---""")
        # st.markdown("## Please draw your molecule(s) in the box below:")
        # editor_column, results_column = st.columns(2)
        similar_smiles = []


        with st.container(border=True):
            st.markdown('## Steps to make predictions')
            st.info("1. Select the properties you want to predict in the sidebar")
            st.info("2. Select the input type in the sidebar")
            st.info("3. Draw your molecule in the box below")
            st.info("4. Click the predict button to get the predicted properties of your molecule")
        # with editor_column:
        with st.expander("Click here to draw your molecule "):
            smiles = st_ketcher(st.session_state.molfile)


        # with results_column:
        with st.expander("Click here to check similar molecules"):
            similarity_threshold = st.slider("Similarity threshold:", min_value=60, max_value=100)
            similar_molecules = utils.find_similar_molecules(smiles, similarity_threshold)
            if not similar_molecules:
                st.warning("No results found")
            else:
                table, similarity_df = utils.render_similarity_table(similar_molecules)
                similar_smiles = utils.get_similar_smiles(similar_molecules)
                st.markdown(f'<div id="" style="overflow:scroll; height:400px; padding-left: 20px;">{table}</div>',
                            unsafe_allow_html=True)

        smile_code = smiles.replace('.',', ')
        molecule = st.sidebar.text_input("SMILES Representation", smile_code)
        moleculesList = molecule.split(",")
        prediction2 = st.sidebar.button('Predict property of molecule', type='primary', use_container_width=True)
        # with results_column:
        with st.expander("Click here to view the 3D structure of your molecule"):
            st.info("Coming soon")
            # if len(moleculesList) >0 and molecule != "":
            #     for  smile in moleculesList:
            #         display_3D_molecule(smile, width=400, height=400)


        if prediction2:

            moleculesList = molecule.split(",")
            df_original = pd.DataFrame(moleculesList, columns=['SMILES'])
            output_df = generate_prediction_dataframe(df_original, desc_df_columns, scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha)
            html_df = display_molecule_in_dataframe_as_html(output_df)
            # with results_column:
            with st.expander("Show predicted properties", expanded=False):
                st.markdown(f'<div id="" style="overflow:scroll; height:400px; padding-left: 20px;">{html_df}</div>',
                                unsafe_allow_html=True)

            molecular_weight = output_df['Molecular Weight'].tolist()[0]

            predicted_Cv = str(output_df['Predicted Cv (cal/mol.K)'].tolist()[0]) + ' cal/mol.K'
            predicted_G = str(output_df['Predicted G (Ha)'].tolist()[0]) + ' kcal/mol'
            predicted_H = str(round(output_df['Predicted G (Ha)'].tolist()[0] - 0.21, 3)) + ' kcal/mol'
            predicted_U = str(output_df['Predicted G (Ha)'].tolist()[0] + 1.25) + ' kcal/mol'
            predicted_mu = str(output_df['Predicted mu (D)'].tolist()[0]) + ' D'
            predicted_homo = str(output_df['Predicted HOMO (Ha)'].tolist()[0]) + ' eV'
            predicted_lumo = str(round(output_df['Predicted LUMO (Ha)'].tolist()[0],4)) + ' eV'
            predicted_bandgap = str(output_df['Predicted LUMO (Ha)'].tolist()[0] - output_df['Predicted HOMO (Ha)'].tolist()[0]) + ' eV'
            predicted_alpha = str(output_df['Predicted alpha (a³)'].tolist()[0]) + ' Å³'


            predicted_property_names = ['Cv', 'G', 'Dipole Moment', 'U', 'H', 'Polarizability', 'Band Gap', 'HOMO', 'LUMO']
            predicted_property_values = [predicted_Cv, predicted_G, predicted_mu, predicted_U, predicted_H, predicted_alpha, predicted_bandgap, predicted_homo, predicted_lumo]


            with st.container(border=True):
                st.markdown("## Predicted Properties")
                structure, properties = st.columns([1,2])

                with structure:
                    with st.container(border=True):
                        st.info('Structure')
                        images = glob.glob('images/*.png')
                        st.image(images)

                    with st.container(border=True):
                        st.info('CHEMBL ID')
                        chembl_id = similarity_df['ChEMBL ID'].tolist()[0]
                        st.markdown(chembl_id, unsafe_allow_html=True)

                with properties:
                    with st.container(border=True):
                        col1, col2, col3 = st.columns(3)
                        for i, col in enumerate([col1, col2, col3]):
                            with col:
                                for j in range(3):
                                    with st.container(border=True):
                                        st.markdown(f'<div id="" style=" height:50px; background-color: #ff4b4b; border-radius: 10px; display: flex; justify-content: center; align-items: center; font-weight: bold">{predicted_property_names[3*i+j]}</div>',
                                                    unsafe_allow_html=True)
                                        #st.success(f'{predicted_property_names[3*i+j]}')
                                        st.markdown(
                                            f'<div id="" style="display: flex; justify-content: center; align-items: center; font-size: 20px; height:100px; ">{predicted_property_values[3 * i + j]}</div>',
                                            unsafe_allow_html=True)

                                        #st.metric(label="Predicted", value=predicted_property_values[3*i+j], delta="", label_visibility='collapsed')

def page_selection():
    selected = sac.segmented(items=[
        sac.SegmentedItem(label='Predictor', icon='🔮' ),
        sac.SegmentedItem(label='Project Overview', icon='🏠')],  format_func='title', align='center', use_container_width=True)

    if selected == "Project Overview":
        st.markdown("""---""")
        project_overview_page.project_overview()

    if selected == "Predictor":
        Prediction()


page_selection()