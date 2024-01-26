import pandas as pd
from scripts.utils import mordred_descriptors_dataframe
from rdkit import Chem
from rdkit.Chem import Descriptors


def predict_property_cv(X_test_scaled, model):
    X_Cv = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_Cv, columns =['Predicted Cv (cal/mol.K)'])
    return predicted
def predict_property_G(X_test_scaled, model):
    X_G = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_G, columns =['Predicted G (Ha)'])
    return predicted
def predict_property_mu(X_test_scaled, model):
    X_mu = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_mu, columns =['Predicted mu (D)'])
    return predicted

def predict_property_homo(X_test_scaled, model):
    X_homo = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_homo, columns =['Predicted HOMO (Ha)'])
    return predicted

def predict_property_lumo(X_test_scaled, model):
    X_lumo = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_lumo, columns =['Predicted LUMO (Ha)'])
    return predicted

def predict_property_alpha(X_test_scaled, model):
    X_alpha = model.predict(X_test_scaled)
    predicted = pd.DataFrame (X_alpha, columns =['Predicted alpha (a³)'])
    return predicted


def generate_prediction_dataframe(df, desc_df_columns, scaler, model_cv, model_G, model_mu, model_homo, model_lumo, model_alpha):
    X_test_scaled = mordred_descriptors_dataframe(df, desc_df_columns, scaler)
    X_Cv = predict_property_cv(X_test_scaled, model_cv)
    X_Cv['Predicted Cv (cal/mol.K)'] = X_Cv['Predicted Cv (cal/mol.K)'].apply(lambda x: round(x, 2))
    X_G = predict_property_G(X_test_scaled, model_G)
    X_G['Predicted G (Ha)'] = X_G['Predicted G (Ha)'].apply(lambda x: round(x, 2))
    X_mu = predict_property_mu(X_test_scaled, model_mu)
    X_mu['Predicted mu (D)'] = X_mu['Predicted mu (D)'].apply(lambda x: round(x, 2))
    X_homo = predict_property_homo(X_test_scaled, model_homo)
    X_homo['Predicted HOMO (Ha)'] = X_homo['Predicted HOMO (Ha)'].apply(lambda x: round(x, 4))
    X_lumo = predict_property_lumo(X_test_scaled, model_lumo)
    X_lumo['Predicted LUMO (Ha)'] = X_lumo['Predicted LUMO (Ha)'].apply(lambda x: round(x, 4))
    X_alpha = predict_property_alpha(X_test_scaled, model_alpha)
    X_alpha['Predicted alpha (a³)'] = X_alpha['Predicted alpha (a³)'].apply(lambda x: round(x, 2))
    df['Molecular Weight'] = [Descriptors.MolWt(mol) for mol in df['mol']]
    output_df = pd.concat([df, X_Cv, X_G, X_mu, X_homo, X_lumo, X_alpha], axis=1)
    output_df.drop(columns=['mol'], inplace=True)
    return output_df