import pandas as pd
import pybnesian as pb
import itertools
import math
import numpy as np
from numpy.random import choice
import random

#Tiene completado de dataset
def EM(red, dataframe, clusters_names, kmax=100):
    k = 0
    rb = red.clone()

    df = dataframe.copy()
    # Instanciamos los parámetros de la variable cluster, como se va a hacer stochastic EM en vez de fijar los parámetros simplemente sampleamos una muestra de la variable cluster
    # para parámetros sampleados de una uniforme.

    probability_distribution = [np.random.uniform() for i in range(len(clusters_names))]
    ini_clust = choice(clusters_names, df.shape[0],p=[x / sum(probability_distribution) for x in probability_distribution])


    df['cluster'] = ini_clust
    # completar dataset
    categor = []
    for var in dataframe.columns:
        categor.append(dataframe[var].cat.categories.tolist())
    categor.append(clusters_names)
    combinations = list(itertools.product(*categor))
    x = pd.DataFrame(combinations, columns=df.columns)
    df = pd.concat([df, x])
    for var in df.columns:
        df[var] = df[var].astype('category')

    # Fit
    # Con esta muestra sampleada hacemos el primer fit de la red utilizando la libreria de pybnesian.
    rb.fit(df)

    # Comenzamos las iteraciones del algoritmo
    while k < kmax:

        new_c = []  # esta lista va a contener la nueva muestra sampleada de P(C|X) en el paso expectation
        logl = []  # esta lista contiene las listas de los loglikelihoods para cada punto del dataframe con cada posible cluster: P(c'k',X)
        # Este loglikelihood se utilizará para el cálculo de P(C|X)=P(C,X)/P(X). Como C es categórica con elevar e al loglikelhood tenemos P(C,X) y normalizando P(C|X)
        df = dataframe.copy()
        for cluster in clusters_names:  # para cada cluster añadimos una columna al dataframe con el cluster dado para poder calcular P(c'k',X)
            c = []
            for i in range(df.shape[0]):
                c.append(cluster)
            df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))

            logl.append(rb.logl(df).tolist())  # Dado el dataframe utilizamos pybnesian para calcular el loglikelihood.

        for row in range(df.shape[0]):  # Para cada dato ahora calculamos P(C|x) y sampleamos de forma que con el nuevo dataframe que construyamos hacemos el fit de la red (cálculo de parámetros)
            # De esta forma estamos ralizando Montecarlo EM con un único sample
            lklh = []  # lista con los loglikelihood para cada cluster
            for n in range(len(clusters_names)):
                x = math.exp(logl[n][row])
                lklh.append(x)
            t = sum(lklh)
            prob_posterior = [x / t for x in lklh]  # normalización
            new_c.append(np.random.choice(np.asarray(clusters_names), p=prob_posterior))
        df['cluster'] = new_c
        # completar dataset
        categor=[]
        for var in dataframe.columns:
            categor.append(dataframe[var].cat.categories.tolist())
        categor.append(clusters_names)
        combinations = list(itertools.product(*categor))
        x = pd.DataFrame(combinations, columns=df.columns)
        df = pd.concat([df, x])
        for var in df.columns:
            df[var] = df[var].astype('category')
        # Fit
        # Con esta muestra sampleada hacemos el fit de la red utilizando la libreria de pybnesian.
        rb.fit(df)
        rb = red.clone()
        rb.fit(df)
        k = k + 1

    return rb


def structure_logl(red, dataframe, clusters_names, sample=20): #esta función estima el expected loglikelihood de los datos.
    posterioris_x = []
    rb = red.clone()
    logl = []  # esta lista contiene las listas de los loglikelihoods para cada punto del dataframe con cada posible cluster: P(c'k',X)
    slogl = []  # esta lista contiene la suma de los loglikelihood de cada punto del dataframe para cada muestra de 'clusters' sampleada de P(C|X)
    df = dataframe.copy()

    for cluster in clusters_names:  # para cada cluster añadimos una columna al dataframe con el cluster dado para poder calcular P(c'k',X)

        c = [cluster for i in range(df.shape[0])]
        df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))
        logl.append(rb.logl(df).tolist())

    for row in range(df.shape[0]):
        lklh = []
        for cluster in range(len(clusters_names)):
            x = math.exp(logl[cluster][row])
            lklh.append(x)
        t = sum(lklh)
        prob_posterior = [x / t for x in lklh]
        posterioris_x.append(prob_posterior)

    for i in range(0, sample):
        new_c = []
        df = dataframe.copy()
        for k in range(df.shape[0]):
            new_c.append(np.random.choice(np.asarray(clusters_names), p=posterioris_x[k]))
        new_c = pd.Series(pd.Categorical(new_c, categories=clusters_names))
        df['cluster'] = new_c

        slogl.append(red.slogl(df))

    return sum(slogl) / len(slogl)


def n_param(red, number_of_clusters,categories_df):  # dado una red, el nº de clusters y las categorías de cada variable (es inaccesible por pybnesian) calculamos el nº de parámetros estimados
    n = number_of_clusters - 1  # comenzamos con el nº de parámetros de la variable cluster que como bien sabemos es number_clusters-1 debido a que trabajamos con un TAN
    for var in red.children('cluster'):  # como todos son hijos de cluster accedemos a las variables de esta forma.
        n = n + (red.num_parents(var)) * (len(categories_df[var]) - 1)

    return n


def sem(bn, dataframe, categories_df, clusters_names, max_iter=2, em_kmax=500, structlog_sample=500):
    clgbn = EM(bn, dataframe, clusters_names, em_kmax)  # comenzamos estimando la red naive bayes
    best = clgbn.clone()
    i = 0  # controla cuando se llega a un punto estacionario el máximo nº de iteraciones a realizar
    df = dataframe.copy()
    BIC = -2 * structure_logl(clgbn, df, clusters_names, structlog_sample) + math.log(df.shape[0]) * n_param(clgbn,len(clusters_names),categories_df)  # bic de la primera red naive


    print(BIC)
    participant_nodes = list(df.columns.copy())  # Nodos de la red
    possible_arcs = list(itertools.permutations(participant_nodes,2))  # Posibles parejas de arcos entre los nodos de la red salvo cluster

    # Comenzamos el algoritmo sem
    while i < max_iter:
        s = 0  # controla si se ha mejorado en la iteración
        k = 0  # lo utilizamos para ir recorriendo la lista de posibles acciones (en este caso añadir arco)
        random.shuffle(possible_arcs)  # mezclamos aleatoriamente los posibles arcos que se pueden introducir
        while k < len(possible_arcs):

            if clgbn.can_add_arc(possible_arcs[k][0], possible_arcs[k][1]):  # comprobamos si el arco puede ser añadido
                red = pb.DiscreteBN(nodes=clgbn.nodes(),arcs=clgbn.arcs())  # en caso de que pueda ser añadido generamos una nueva red con el arco añadido, estimamos parámetros y comparamos BIC
                red.add_arc(possible_arcs[k][0], possible_arcs[k][1])
                red = EM(red, df, clusters_names, em_kmax)
                l = -2 * structure_logl(red, df, clusters_names, structlog_sample) + math.log(df.shape[0]) * n_param(red, len(clusters_names), categories_df)

                if l >= BIC:  # si no mejoramos pasamos al siguiente arco
                    k = k + 1
                else:  # si mejoramos BIC se actualiza y red se actualiza
                    k = len(possible_arcs)
                    BIC = l
                    clgbn = red
                    best = clgbn.clone()
                    s = s + 1
            else:
                k = k + 1  # si no se puede introducir el arco pasamos al siguiente

        k = 0
        possible = list(
            clgbn.arcs())  # En este caso necesitamos trabajar con la lista de arcos existentes ya que tratamos de invertir alguno

        for element in participant_nodes:  # eliminamos los arcos (cluster,variable) ya que estos no deben tocarse
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)  # hacemos un random order de los posibles candidatos

        while k < len(possible):
            if clgbn.can_flip_arc(possible[k][0], possible[k][1]):  # si se puede invertir el arco de nuevo probamos y comparamos si se mejora el BIC
                red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
                red.flip_arc(possible[k][0], possible[k][1])
                red = EM(red, df, clusters_names, em_kmax)
                l = -2 * structure_logl(red, df, clusters_names, structlog_sample) + math.log(df.shape[0]) * n_param(red, len(clusters_names), categories_df)


                if l >= BIC:
                    k = k + 1
                else:
                    k = len(possible)
                    BIC = l
                    clgbn = red
                    best = clgbn.clone()
                    s = s + 1
            else:
                k = k + 1

        k = 0
        possible = list(
            clgbn.arcs())  # de nuevo queremos eliminar arcos por ello los candidatos son los arcos ya existentes salvo los que no se deben eliminar (cluster,variable)
        for element in participant_nodes:
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)

        while k < len(possible):
            red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
            red.remove_arc(possible[k][0], possible[k][1])
            red = EM(red, df, clusters_names, em_kmax)
            l = -2 * structure_logl(red, df, clusters_names, structlog_sample) + math.log(df.shape[0]) * n_param(red,len(clusters_names),categories_df)

            if l >= BIC:
                k = k + 1
            else:
                k = len(possible)
                BIC = l
                clgbn = red
                best = clgbn.clone()
                s = s + 1

        print(BIC)
        if s == 0:  # si no se mejora comienza el contador de maxiter
            i = i + 1
        else:
            i = 0

    return best

