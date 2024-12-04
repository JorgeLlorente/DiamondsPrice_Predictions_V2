import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import support as src
from sklearn.model_selection import train_test_split
from sklearn import metrics


def visual_categ(df):

    """
    Def:
        Calcula la distribución de las variables categóricas.

    Args:
        df: un dataframe de pandas.
    
    """

    col_categ = df.select_dtypes(include="object").columns
    num_col_categ = len(df.select_dtypes(include="object").columns)
    colors = ["green", "skyblue", "purple"]

    fig, axes = plt.subplots(nrows= (2 if num_col_categ >= 4 else 1), ncols= num_col_categ, figsize = (30,10))

    axes = axes.flat


    for i, column in enumerate(col_categ):

        value_counts = df[column].value_counts()
        axes[i].bar(value_counts.index, value_counts.values, color = colors[i])
        axes[i].set_title(column)
        axes[i].tick_params(axis = "y", labelsize = 20)
        axes[i].tick_params(axis = "x", labelsize = 20, rotation=-45)


def mean_price_categ(df):

    """
    Def:
        Calcula el promedio del precio para cada valor de cada categoría

    Args:
        df: un dataframe de pandas.
    
    """

    col_categ = df.select_dtypes(include="object").columns
    num_col_categ = len(df.select_dtypes(include="object").columns)
    colors = ["green", "skyblue", "purple"]

    fig, axes = plt.subplots(nrows= (2 if num_col_categ >= 4 else 1), ncols= num_col_categ, figsize = (30,10))

    axes = axes.flat

    for i, column in enumerate(col_categ):
        agg_data = df.groupby(column)["price"].mean()
        df_cat = pd.DataFrame(agg_data)
        df_cat.reset_index(inplace=True)
        axes[i].bar(df_cat[column], height=df_cat["price"], color=colors[i])
        axes[i].set_title(column)


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):

    """
    Def:
        Dividimos el conjunto de datos en entrenamiento, validacióno y prueba.
    
    Args:
        df: un dataframe de pandas
    """

    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):

    """
    Def:
        Separamos las características de la variable respuesta.

    Args:
        df: un dataframe de pandas.
        label_name: nombre de la columna de la variable respuesta.
    """

    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)


def overfit_model(y_train, y_train_pred, y_val, y_val_pred):

    """"
    Def:
        Comprobar si el modelo tiene overfitting usando la métrica R^2.
    
        Args:
        y_train: array con los valores de la variable de salida.
        y_train_pred: array con los valores predecidos para el conjunto de datos de entrenamiento.
        y_val: array con los valores de la variable de salida del set de validación.
        y_val_pred: array con los valores predecidos para el conjunto de validación.
    """
    
    train_r2 = metrics.r2_score(y_train, y_train_pred)
    val_r2 = metrics.r2_score(y_val, y_val_pred)
    print(f"R^2 en entrenamiento: {train_r2}")
    print(f"R^2 en validación: {val_r2}")
