
import glob # for image handling
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from chembl_webresource_client.new_client import new_client as ch # for similarity search
from rdkit import Chem
from rdkit.Chem import AllChem # for molecule handling
from rdkit.Chem import Draw # for molecule drawing
# import py3Dmol
# from stmol import showmol
from mordred import Calculator, descriptors # for descriptor calculation
from scripts.image_handling import convert_df

from py2opsin import py2opsin

# from st_speckmol import speck_plot
EBI_URL = "https://www.ebi.ac.uk/chembl/"

# def display_3D_molecule(smiles, width, height):
#     mol = Chem.MolFromSmiles(smiles)
#     mol = Chem.AddHs(mol)
# def display_3D_molecule(smiles, width, height):
#     mol = Chem.MolFromSmiles(smiles)
#     mol = Chem.AddHs(mol)
#     AllChem.EmbedMolecule(mol)
#     AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
#     mblock = Chem.MolToMolBlock(mol)
#     view = py3Dmol.view(width=width, height=height)
#     view.addModel(mblock, 'mol')
#     view.setStyle({"stick": {}})
#     view.zoomTo()
#     view.spin(spin_on=True)
#     showmol(view)
#


def get_smiles_from_name(name):
    try:
        smiles_string = py2opsin(
            chemical_name = name,
            output_format = "SMILES",)
        return smiles_string
    except Exception as _:
        print("No smiles string found")
        return None
def display_molecule_in_dataframe_as_html(dataframe):
    df = dataframe
    for index, i in enumerate(df['SMILES']):
        Draw.MolToFile(Chem.MolFromSmiles(str(i)), f'images/{index}{i}.png',   size=(300, 300), fitImage=True, imageType='png')

    images = glob.glob('images/*.png')
    df['Structure'] = images
    html_df = convert_df(df)
    return html_df


def display_molecule(molecule): # Function to display molecules
    img = Draw.MolToImage(molecule, size=(1000, 1000), fitImage=True)
    st.image(img)

def display_molecule_dataframe(dataframe):
    df = dataframe
    #img = Draw.MolsToGridImage([Chem.MolFromSmiles(str(i)) for i in df['SMILES']], molsPerRow=2, subImgSize=(300, 300), legends=[str(j +'\n' + '\n' + str(round(i, 2)) + " J/mol.K") for i, j in zip(df['Predicted Cv (J/mol.K)'], df['SMILES'])])

    img = Draw.MolsToGridImage([Chem.MolFromSmiles(str(i)) for i in df['SMILES']], molsPerRow=2, subImgSize=(300, 300), legends=[str(j) for i, j in zip(df['Predicted Cv (cal/mol.K)'], df['SMILES'])])
    st.image(img, use_column_width=True)
    images = []
    for i in df['SMILES']:
        img = Draw.MolToFile(Chem.MolFromSmiles(str(i)), f'static/{i}.png',   size=(300, 300), fitImage=True, imageType='png')
        images.append(img)
    df['Image'] = images

def canonize(mol):
    return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

def mordred_descriptors(smiles, desc_df_columns, scaler):
    df = pd.DataFrame(smiles, columns=['SMILES'])
    canonsmiles = [canonize(str(i)) for i in smiles]
    mols = [Chem.MolFromSmiles(str(i)) for i in canonsmiles]
    df['mol'] = mols
    calc = Calculator(descriptors, ignore_3D=True)
    df_desc = calc.pandas(df['mol'])
    df_desc = df_desc.select_dtypes(include=np.number).astype('float32')
    df_desc['ABC'] = 0
    df_desc['ABCGG'] = 0
    df_desc = df_desc[desc_df_columns]
    df_descN = pd.DataFrame(scaler.transform(df_desc), columns=df_desc.columns)
    print(df_descN)
    return df_descN

def mordred_descriptors_dataframe(dataframe, desc_df_columns, scaler):
    df = dataframe
    canonsmiles = [canonize(str(i)) for i in df['SMILES']]
    df['mol'] = [Chem.MolFromSmiles(str(i)) for i in canonsmiles]
    calc = Calculator(descriptors, ignore_3D=True)
    df_desc = calc.pandas(df['mol'])
    df_desc = df_desc.select_dtypes(include=np.number).astype('float32')
    df_desc['ABC'] = 0
    df_desc['ABCGG'] = 0
    df_desc = df_desc[desc_df_columns]
    #df_desc.to_csv('df_desc1.csv')
    df_descN = pd.DataFrame(scaler.transform(df_desc), columns=df_desc.columns)
    print(df_descN)
    #df_descN.to_csv('df_descN1.csv')
    return df_descN



def name_to_molecule(name: str) -> Tuple[str, str]:
    columns = ['molecule_chembl_id', 'molecule_structures']
    ret = ch.molecule.filter(molecule_synonyms__molecule_synonym__iexact=name).only(columns)
    best_match = ret[0]
    return best_match["molecule_structures"]["molfile"], best_match["molecule_chembl_id"]


def id_to_molecule(chembl_id: str) -> Tuple[str, str]:
    return ch.molecule.filter(chembl_id=chembl_id).only('molecule_structures')[0]["molecule_structures"]["molfile"]


def style_table(df: pd.DataFrame):
    return df.style.format(
        subset=['Similarity'],
        decimal=',', precision=2
    ).bar(
        subset=['Similarity'],
        align="mid",
        cmap="coolwarm"
    ).applymap(lambda x: 'background-color: #aaaaaa', subset=['Image'])


def style_predictions(df: pd.DataFrame):
    return df.style.format(
        subset=['Prediction'],
        decimal=',', precision=2
    ).bar(
        subset=['Prediction'],
        align="mid",
        cmap="plasma_r",
        vmax=1.0,
        vmin=0.8
    )


def render_chembl_url(chembl_id: str) -> str:
    return f'<a href="{EBI_URL}compound_report_card/{chembl_id}/">{chembl_id}</a>'


def render_chembl_img(chembl_id: str) -> str:
    return f'<img src="{EBI_URL}api/data/image/{chembl_id}.svg" height="100px" width="100px">'


def render_row(row):
    return {
        "Similarity": float(row["similarity"]),
        "Preferred name": row["pref_name"],
        "ChEMBL ID": render_chembl_url(row["molecule_chembl_id"]),
        "Image": render_chembl_img(row["molecule_chembl_id"])
    }


def render_target(target):
    return {
        "Prediction": float(target["pred"]),
        "ChEMBL ID": render_chembl_url(target["chembl_id"])
    }


def find_similar_molecules(smiles: str, threshold: int):
    columns = ['molecule_chembl_id', 'similarity', 'pref_name', 'molecule_structures']
    try:
        return ch.similarity.filter(smiles=smiles, similarity=threshold).only(columns)
    except Exception as _:
        return None


def render_similarity_table(similar_molecules):
    records = [render_row(row) for row in similar_molecules if row["molecule_structures"]]
    df = pd.DataFrame.from_records(records)
    styled = style_table(df)
    return styled.to_html(render_links=True), df  # return the dataframe as well


def render_target_predictions_table(predictions) -> Optional[str]:
    df = pd.DataFrame(predictions)
    records = [render_target(target) for target in
               df.sort_values(by=['pred'], ascending=False).head(20).to_dict('records')]
    df = pd.DataFrame.from_records(records)
    styled = style_predictions(df)
    return styled.to_html(render_links=True)


def get_similar_smiles(similar_molecules):
    return [mol["molecule_structures"]["canonical_smiles"] for mol in similar_molecules if mol["molecule_structures"]]