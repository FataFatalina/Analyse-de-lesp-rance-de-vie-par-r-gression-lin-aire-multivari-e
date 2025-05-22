# Ici on trouve toutes les fonction utilisées dans le projet **** life_expectancy_project****

# Fonctions pour l'analyse univariée 
#--------------------------------------------------------------------------------------------#

def split_time_series(df, date_col='Year', train_size=0.8):
    """
    Split a time series dataframe into train, validation, and test sets while keeping chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        The time series dataframe sorted by date.
    date_col : str
        The name of the column containing dates.
    train_size : float
        Proportion of data to use for training.


    Returns
    -------
    tuple : (df_train, df_test)
        Chronologically split datasets.
    """
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_size)
   

    df_train = df_sorted.iloc[:train_end]
    df_test = df_sorted.iloc[train_end:]

    return df_train, df_test


# How to use: 

# df_train, df_test = split_time_series(df)


#--------------------------------------------------------------------------------------------#
# Fonction to see the missing % of data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_data(df: pd.DataFrame):
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_percent.values, y=missing_percent.index, palette='coolwarm')
    plt.title('Proportion de données manquantes (%)')
    plt.xlabel('% de valeurs manquantes')
    plt.ylabel('Colonnes')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

# How to use: 

# plot_missing_data(df_train)


#--------------------------------------------------------------------------------------------#
# Outlier reaserching function
    # Fonction de creation de boxplot pour chaque donnée numérique
def plot_box_donnees_numeriques(df: pd.DataFrame, features):
    dfnbr = [f for f in df.columns if df.dtypes[f] != 'object']
    f = pd.melt(df, value_vars=features)
    g = sns.FacetGrid(f, col="variable", col_wrap=5, sharex=False, sharey=False, height=5)
    g.map(sns.boxplot, "value")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Distribution des Attributs Numériques")
    plt.show()
# Usage :
# plot_box_donnees_numeriques(X_train, num_features)



#--------------------------------------------------------------------------------------------#
# Function that calculates the proportion of outliers proportion in the Dataframe
def proportion_valeurs_abberantes(df):
    outlier_info = {}
    for column in df.select_dtypes(include=['float64', 'int64']): 
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        total_count = df[column].count()
        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].count()
        outlier_percentage = (outlier_count / total_count) * 100
        outlier_info[column] = outlier_percentage
    return outlier_info


# Usage :percentag_abberantes = proportion_valeurs_abberantes(X_train)
# print("Pourcentage des valeurs abbéranres pour chaque variable:")
# for variable, percentage in percentag_abberantes.items():
#     print(f"{variable}: {percentage:.2f}%")

#--------------------------------------------------------------------------------------------#
# Function to plot the outliers proportion in the dateframe

import matplotlib.pyplot as plt
import seaborn as sns

def plot_outlier_proportions(outlier_info):
    # Transformer le dictionnaire en DataFrame pour trier et tracer facilement
    df_outliers = pd.DataFrame.from_dict(outlier_info, orient='index', columns=['% de valeurs aberrantes'])
    df_outliers = df_outliers.sort_values(by='% de valeurs aberrantes', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='% de valeurs aberrantes', y=df_outliers.index, data=df_outliers, palette='rocket')
    plt.title('Proportion de valeurs aberrantes par variable')
    plt.xlabel('% de valeurs aberrantes')
    plt.ylabel('Variables')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



#--------------------------------------------------------------------------------------------#
# Function to impute the missing values

def groupwise_impute(X, grouping_cols: dict):
    """
    Remplit les valeurs manquantes dans X selon une imputation par groupe.
    
    Parameters:
    - X : pandas DataFrame
    - grouping_cols : dictionnaire {colonne: groupe} indiquant pour chaque colonne
                      sur quelle variable regrouper pour faire l’imputation.
    
    Returns:
    - X_imputed : une copie de X avec les valeurs manquantes imputées.
    """
    X = X.copy()
    for col, group_col in grouping_cols.items():
        X[col] = X.groupby(group_col)[col].transform(lambda x: x.fillna(x.mean()))
    return X


# Création of a dictionary containing the colums that are grouped and by whcih colums they are grouped
# grouping_dict = {
#     # 'Prevelance of Undernourishment': 'Region',
#     # 'CO2': 'Region',
#     # 'Health Expenditure %': 'IncomeGroup',
#     # 'Education Expenditure %': 'IncomeGroup',
#     # 'Unemployment': 'Region',	
#     # 'Corruption': 'Region',
#     # 'Sanitation': 'Region',
#     'Life Expectancy World Bank': 'IncomeGroup'
# }

# Usage :

# X_train_imputed = groupwise_impute(X_train, grouping_dict)
# X_test_imputed = groupwise_impute(X_test, grouping_dict)



#--------------------------------------------------------------------------------------------#

# Function to sacve data to csv files
def save_to_csv(data, filename):
    """
    Saves a pandas Series or DataFrame to a CSV file with index.

    Parameters:
    - data : pd.Series or pd.DataFrame — The data to save
    - filename : str — The filename (or path) to save to, e.g. 'output.csv'
    """
    import pandas as pd

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data.to_csv(filename, index=True)
    else:
        raise ValueError("Input must be a pandas Series or DataFrame")
    

# Usage :
# save_to_csv(y_train_, "y_train.csv")
# save_to_csv(X_train_imputed, "X_train.csv")

        


#--------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
# This function extracts times series dataframe, we get a dictionary of  dataframes where the keys are the different regions . 
# The actual data contains many values for the same region because a region is composed of many countries and ther i a value of life expectancy per contry, but as our analysisi is focused regionaly, we take the average of the life expectancy per region.

def extract_timeseries_by_region(df, region_list, target_col='Life Expectancy World Bank',
                                  year_col='Year', region_col='Region'):
    """
    Extracts and aggregates time series data for each region in region_list.

    For each region:
    - Filters the data by region
    - Drops missing values
    - Groups by year and computes the average of the target variable
    - Sorts by year

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset containing year, region, and target columns.
    region_list : list of str
        List of regions to extract.
    target_col : str
        Name of the target variable to extract.
    year_col : str
        Name of the column containing years.
    region_col : str
        Name of the column containing region identifiers.

    Returns
    -------
    dict
        A dictionary with region names as keys and corresponding DataFrames as values,
        each with two columns: Year and target_col (averaged), sorted by Year.
    """
    timeseries_dict = {}

    for region in region_list:
        # Filter for the region and remove missing values
        df_region = df[df[region_col] == region][[year_col, target_col]].dropna()

        if df_region.empty:
            continue

        # Group by year and calculate the average value for the target column
        df_region_agg = (
            df_region.groupby(year_col, as_index=False)
            .mean()
            .sort_values(by=year_col)
        )

        timeseries_dict[region] = df_region_agg.reset_index(drop=True)

    return timeseries_dict



#--------------------------------------------------------------------------------------------#
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def appliquer_prophet_sur_dictionnaire(dict_df, annee_future=4):
    resultats = {}
    nb_regions = len(dict_df)
    
    fig, axes = plt.subplots(1, nb_regions, figsize=(6 * nb_regions, 5), constrained_layout=True)

    if nb_regions == 1:
        axes = [axes]  # pour rendre iterable si une seule région

    for i, (region, df) in enumerate(dict_df.items()):
        print(f" Traitement de la région : {region}")
        
        # Étape 1 : préparation des données
        data = df.copy()
        data.columns = ['ds', 'y']
        data['ds'] = pd.to_datetime(data['ds'], format="%Y")
        
        # Étape 2 : création et entraînement du modèle
        model = Prophet()
        model.fit(data)

        # Étape 3 : création du futur
        future = model.make_future_dataframe(periods=annee_future, freq='YE')
        future = future.drop_duplicates(subset='ds')

        # Étape 4 : prédictions
        forecast = model.predict(future)
        resultats[region] = forecast

        # Étape 5 : visualisation sur subplot
        model.plot(forecast, ax=axes[i])
        axes[i].set_title(f"{region}")
        axes[i].set_xlabel("Année")
        axes[i].set_ylabel("Valeur prédite")
        axes[i].grid(True)

    plt.suptitle("Prévision de l'espérance de vie par région", fontsize=16)
    plt.show()

    return resultats





#--------------------------------------------------------------------------------------------#
def comparer_forecasts(dict_forecast, dict_test):
    dict_comparaison = {}

    for region in dict_forecast:
        print(f"Traitement de la région : {region}")
        
        # Récupérer les prédictions du modèle Prophet
        forecast = dict_forecast[region].copy()
        forecast['ds'] = forecast['ds'].dt.year
        forecast = forecast.drop_duplicates(subset='ds')
        #forecast = forecast.set_index('ds')

        # Récupérer les données de test
        test_df = dict_test[region].copy()
        #test_df = test_df.set_index('ds')

        # Garder uniquement les années communes entre test et prédiction
        years_communes = test_df.index.intersection(forecast.index)

        # Comparaison : créer le dataframe avec valeurs réelles et prédites
        comparaison = pd.DataFrame({
            'ds' : test_df.loc[years_communes, 'ds'].values,
            'Y_reel': test_df.loc[years_communes, 'y'].values,
            'Y_hat': forecast.loc[years_communes, 'yhat'].values
        }, index=years_communes)

        dict_comparaison[region] = comparaison

    return dict_comparaison

#--------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import math

def plot_comparison_dict(comparison_dict):
    n = len(comparison_dict)
    cols = 4  # 2 plots per row
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 6, rows * 4))

    for i, (region, comparison) in enumerate(comparison_dict.items(), 1):
        plt.subplot(rows, cols, i)
        plt.plot(comparison['ds'], comparison['Y_reel'], label='Valeurs réelles', marker='o')
        plt.plot(comparison['ds'], comparison['Y_hat'], label='Valeurs prédites', marker='x', linestyle='--')

        plt.title(f'{region}')
        plt.xlabel('Année')
        plt.ylabel('Life Expectancy')
        plt.legend()
        plt.grid(True)

        # Ensure the X-axis shows years as integers
        plt.gca().set_xticks(comparison['ds'])
        plt.gca().set_xticklabels(comparison['ds'].astype(int))

    plt.tight_layout()
    plt.show()


#--------------------------------------------------------------------------------------------#
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculer_performances_par_region(donnees_par_region):
    """
    Calcule MAE, RMSE et R² pour chaque région.

    Paramètres :
        donnees_par_region (dict): Dictionnaire où chaque clé est une région,
                                   et chaque valeur est un DataFrame ou un dictionnaire
                                   avec les colonnes ou clés 'Réel' et 'Prédit'.

    Retour :
        dict: Dictionnaire des performances par région.
    """
    resultats = {}

    for region, data in donnees_par_region.items():
        # Si c'est un dict, on convertit en arrays
        reel = data['Y_reel']
        predit = data['Y_hat']

        mae = mean_absolute_error(reel, predit)
        rmse = np.sqrt(mean_squared_error(reel, predit))
        r2 = r2_score(reel, predit)

        resultats[region] = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 2)
        }

    return resultats



#--------------------------------------------------------------------------------------------#
# for ARIMA model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

def appliquer_arima_sur_dictionnaire(dict_df_train, dic_df_test ):
    resultats = {}
    summary_dict = {}
    mae = {}
    rmse  = {}
    r2 ={}
    ts_test = dic_df_test['Middle East & North Africa']

    for i, (region, df) in enumerate(dict_df_train.items()):
        print(f" Traitement de la région : {region}")
        
        # Étape 1 : préparation des données
        ts_train = df.copy()
        ts_train = ts_train.set_index('Year')

        ts_test = dic_df_test['Middle East & North Africa'].set_index('Year')
        # Étape 2 : création et entraînement du modèle
        model_arima = ARIMA(ts_train, order=(1,1,1))
        model_arima_fit = model_arima.fit()
        
        summary_dict[region] = model_arima_fit.summary()  # résumé du modèle ARIMA
        
        

        # Prédiction sur la période de test
        n_test = len(ts_test)
        forecast_arima = model_arima_fit.forecast(steps=n_test)
        forecast_arima.index = ts_test.index
        

        # Étape 4 : prédictions
        resultats[region] = forecast_arima
        
        #calcule de metriques de performance pour les predictions de chaque serie temporelle 
        mae[region] = mean_absolute_error(ts_test, forecast_arima)
        rmse[region] = np.sqrt(mean_squared_error(ts_test, forecast_arima))
        r2[region] = r2_score(ts_test, forecast_arima)

        

    return resultats, summary_dict, mae, rmse, r2


#Usage :
#arima_results, summary_dict, mae, rmse, r2 = appliquer_arima_sur_dictionnaire(ts_by_region_train,ts_by_region_test)

#--------------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import math

def plot_previsions_arima_sur_dictionnaire(resultats_arima, dic_df_test):


    n = len(dic_df_test)
    cols = 4  # 2 plots per row
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 6, rows * 4))
    """
    Affiche les prévisions ARIMA et les vraies valeurs (test) pour chaque région.

    Paramètres :
    - resultats_arima : dict[str, pd.Series]
        Prédictions ARIMA par région, chaque valeur est une Series indexée par l’année.
    - dic_df_test : dict[str, pd.DataFrame]
        Vraies valeurs par région. Chaque DataFrame contient les colonnes ['Year', 'Life Expectancy World Bank'].
    """

    for i, (region, forecast_arima) in enumerate(resultats_arima.items(), 1):
        
        if region not in dic_df_test:
            print(f" Données test manquantes pour la région : {region}")
            continue

        df_test = dic_df_test[region]

        ts_test = df_test.set_index("Year")["Life Expectancy World Bank"]

     

        # Vérifier l'alignement des index (années)
        if not ts_test.index.equals(forecast_arima.index):
            print(f" Index non alignés pour la région : {region}")
            print(f"    Index ts_test : {ts_test.index}")
            print(f"    Index forecast : {forecast_arima.index}")
            continue

        # Tracer
        plt.subplot(rows, cols, i)
        plt.plot(df_test['Year'], ts_test.values, label="Valeurs réelles (test)", color='green', marker='o')
        plt.plot(forecast_arima.index, forecast_arima.values, label="Prévisions ARIMA", color='red', linestyle='--', marker='x')

        plt.title(f"Prévisions ARIMA(1,1,1) vs Réalité - {region}")
        plt.xlabel("Année")
        plt.ylabel("Espérance de vie (World Bank)")
        plt.legend()
        plt.grid(True)

         # Ensure the X-axis shows years as integers
        plt.gca().set_xticks(df_test['Year'])
        plt.gca().set_xticklabels(df_test['Year'].astype(int))


    
    plt.tight_layout()
    plt.show()


#Usage:
#plot_previsions_arima_sur_dictionnaire(arima_results, ts_by_region_test)
    
#--------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#