import pyAgrum as gum
import pandas as pd
import itertools

#Creamos la red
bn=gum.BayesNet('Earthquake')


#Añadimos los nodos
bn.add('Alarm', 2)
bn.add('Earthquake', 2)
bn.add('Burglary', 2)
bn.add('JohnCalls', 2)
bn.add('MaryCalls', 2)
#Añadimos los arcos
bn.addArc('Burglary','Alarm')
bn.addArc('Earthquake','Alarm')
bn.addArc('Alarm','JohnCalls')
bn.addArc('Alarm','MaryCalls')
#Añadimos las cpt que son obtenidas del modelo original de Bnlearn https://www.bnlearn.com/bnrepository/discrete-small.html#earthquake
bn.cpt('Burglary').fillWith([0.99,0.01])
bn.cpt('Earthquake').fillWith([0.98,0.02])
bn.cpt("Alarm")[{'Burglary': 0, 'Earthquake': 0}] = [0.999, 0.001]
bn.cpt("Alarm")[{'Burglary': 1, 'Earthquake': 0}] = [0.06, 0.94]
bn.cpt("Alarm")[{'Burglary': 0, 'Earthquake': 1}] = [0.710,0.290]
bn.cpt("Alarm")[{'Burglary': 1, 'Earthquake': 1}] = [0.05, 0.95]
bn.cpt("JohnCalls")[{'Alarm': 0}] = [0.95, 0.05]
bn.cpt("JohnCalls")[{'Alarm': 1}] = [0.1, 0.9]
bn.cpt("MaryCalls")[{'Alarm': 0}] = [0.99, 0.01]
bn.cpt("MaryCalls")[{'Alarm': 1}] = [0.3, 0.7]

#Creamos las estructuras necesarias para hacer inferencia con PyAgrum
variables=['Alarm','Earthquake','JohnCalls','MaryCalls']
ie=gum.LazyPropagation(bn)
ie2=gum.LazyPropagation(bn)

#Definimos in maxgbf inicial para ir guardando los valores máximos
maxgbf=0

#Creamos todas las combinaciones posibles de valores de las variables salvo la evidencia
combinations=[]
for i in range(1,len(variables)+1):
    another_combination=list(itertools.combinations(variables, i))
    combinations.extend(another_combination)
#Fijamos la evidencia
ie.setEvidence({'Burglary':1})

#Para cada combinación calculamos el GBF y comparamos con el máximo guardando la combinación en caso de que se supere, esto finalmente nos dará el MRE
for combination in combinations:
    if len(combination)==1:
        ie.eraseAllMarginalTargets()
        ie.eraseAllJointTargets()
        ie.addTarget(combination[0])
        ie.makeInference()
        cpt=ie.posterior(combination[0]).tolist()
        for q in [0,1]:
            p=cpt[q]
            ie2.eraseAllEvidence()
            dict={combination[0]:q}
            ie2.setEvidence(dict)
            ie2.makeInference()
            ev_p = ie2.evidenceProbability()

            gbf = (p * (1 - ev_p)) / (ev_p * (1 - p))

            if gbf > maxgbf:
                maxgbf = gbf
                max_instantiation = [combination,q]
                max_cpt = cpt


    if len(combination)>1:
        ie.eraseAllJointTargets()
        ie.eraseAllMarginalTargets()
        ie.addJointTarget(set(combination))
        ie.makeInference()
        cpt=ie.jointPosterior(set(combination)).topandas()
        values=cpt.index.values
        for instantiation in values:
            cpt2=cpt.loc[instantiation]
            for r in [0,1]:
                p=cpt2.iloc[r]
                ie2.eraseAllEvidence()
                dict={pd.MultiIndex.from_frame(cpt).names[0][0]:r}
                for t in range(0,len(cpt.index.names)):
                    dict[cpt.index.names[t]]=instantiation[t]
                ie2.setEvidence(dict)
                ie2.makeInference()
                ev_p=ie2.evidenceProbability()
                gbf=(p*(1-ev_p))/(ev_p*(1-p))
                if gbf>maxgbf:
                    maxgbf=gbf
                    max_instantiation=dict
                    max_cpt=cpt


print(max_instantiation)
print(max_cpt)
print(maxgbf)