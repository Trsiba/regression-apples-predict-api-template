"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    

    feature_vector_df = feature_vector_df[(feature_vector_df['Commodities'] == 'APPLE GOLDEN DELICIOUS')]
    
    predict_vector =             feature_vector_df[['Weight_Kg','Low_Price','High_Price','Sales_Total',	'Total_Qty_Sold','Total_Kg_Sold','Stock_On_Hand','Month','Province_EASTERN_CAPE',	'Province_NATAL',	'Province_ORANGE_FREE_STATE',	'Province_TRANSVAAL',	'Province_W.CAPE-BERGRIVER_ETC',	'Province_WEST_COAST',	'Container_DT063',	'Container_EC120',	'Container_EF120',	'Container_EG140',	'Container_JE090',	'Container_JG110',	'Container_M4183',	'Container_M6125',	'Container_M9125',	'Size_Grade_1M',	'Size_Grade_1S',	'Size_Grade_1U',	'Size_Grade_1X',	'Size_Grade_2L',	'Size_Grade_2M',	'Size_Grade_2S',	'Size_Grade_2U',	'Size_Grade_2X',	'Season_Spring',	'Season_Summer',	'Season_Winter']]
                                
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
