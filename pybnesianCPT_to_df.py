import pandas as pd
import pybnesian as pb



def from_CPT_to_df(data_string): #con esta función obtenemos un dataframe con las probabilidades de una variable condicionada a sus padres a partir del string proporcionado
    #por Pybnesian
    # Separamos el string en lineas
    lines = data_string.strip().split('\n')

    # Quitamos los guiones
    lines = [line for line in lines if not line.startswith('+')]



    # Extraemos los nombres de las columnas
    header_separator = 3
    columns = [column.strip() for column in lines[header_separator - 1].split('|') if column.strip()]


    # Extraemos las filas
    data_start = header_separator
    data_rows = [line.split('|')[1:-1] for line in lines[data_start:]]

    # Formateamos los datos
    data = []
    for row in data_rows:
        clean_row = [item.strip() for item in row]
        data.append(clean_row)

    df = pd.DataFrame(data, columns=columns)
    return df