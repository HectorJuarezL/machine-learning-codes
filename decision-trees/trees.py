from math import log
import operator

def createDataSet(): #Funcion que retorna un mequeño dataset
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet): #Esta funcion calcula la entropia de shannon del dataset
    numEntries = len(dataSet) #Obtiene el numero de filas del dataset
    labelCounts = {} #Diccionario donde se guarda la cuenta de cada clase
    for featVec in dataSet: #Obtiene los elementos únicos con sus caracteristicas
        currentLabel = featVec[-1] #Establece currentLable al valor del ultimo elemento del vector de caracteristicas
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 #Comprueba si ya se encuentra la clase actual en la variable de labelCounts, en caso de que no, añade la key al diccionario y la establece en 0
        labelCounts[currentLabel] += 1 #aumenta en uno el valor de la clase 
    shannonEnt = 0.0 #Inicializa la entropia en ceros
    for key in labelCounts: #En este for calcula la probabilidad de cada una de las etiquetas y lo añade a la "suma"
        prob = float(labelCounts[key])/numEntries #aqui calcula la probabilidad
        shannonEnt -= prob * log(prob,2) #aqui calcula la entropia y la añade a la suma, utiliza logaritmo base 2
    return shannonEnt #retorna la entropia
    
def splitDataSet(dataSet, axis, value): #Funcion que retorna todas las filas del dataset cuya columna (axis) sea igual al valor dado
    retDataSet = [] #variable donde se retornará la division
    for featVec in dataSet: #for que recorre cada uno de las filas del dataset
        if featVec[axis] == value: #comprueba si cumple con la condicion dada como parametro
            reducedFeatVec = featVec[:axis]     #crea una fila con todos los datos hasta antes del axis
            reducedFeatVec.extend(featVec[axis+1:]) #añade los elementos restantes menos la columna del axis
            retDataSet.append(reducedFeatVec) #agrega la fila a la matriz de retorno
    return retDataSet #retorna todos aquellos datos segun el valor de la columna dado
    
def chooseBestFeatureToSplit(dataSet): #Esta es la funcion nucleo del algoritmo
    numFeatures = len(dataSet[0]) - 1  #Obtiene el numero de caracteristicas (columnas). Es importante recalcar que la ultima columna pertenece a la etiqueta de clase
    baseEntropy = calcShannonEnt(dataSet) #calcula la entropia completa del dataset
    bestInfoGain = 0.0; bestFeature = -1 #inicializa los valores para la mejor ganancia y la mejor caracteristica
    for i in range(numFeatures):        #for que recorre todas las caracteristicas
        featList = [example[i] for example in dataSet]#Crea una lista con todas las filas que tienen la caracteristica "i"
        uniqueVals = set(featList) #Obtiene un conjunto con todos los valores únicos para esa caracteristica dada
        newEntropy = 0.0 #Inicializa el valor de la entropia
        for value in uniqueVals: #For que recorre todos los valores únicos de la caracteristica "i"
            subDataSet = splitDataSet(dataSet, i, value) #Obtiene el subconjunto de datos con la caracteristica i
            prob = len(subDataSet)/float(len(dataSet)) #Obtiene la probabilidad de este subconjunto
            newEntropy += prob * calcShannonEnt(subDataSet) #Suma la entropia dada para este subconjunto 
        infoGain = baseEntropy - newEntropy     #Calcula la ganancia
        if (infoGain > bestInfoGain):       #Compara la ganancia con la de mejor ganancia
            bestInfoGain = infoGain         #Si es mejor que la de la variable, asigna esta como la nueva mejor
            bestFeature = i #Tambien guarda el valor de i, es decir, el índice de la mejor caracteristica
    return bestFeature #Al final, retorna el índice de la mejor caracteristica.

def majorityCnt(classList): #Funcion que retorna el valor de la clase mayoritaria
    classCount={} #Diccionario donde se guardará la cuenta de cada clase
    for vote in classList: #For que recorre la lista de clases
        if vote not in classCount.keys(): classCount[vote] = 0 #En caso de que no exista la key, la añade como cero
        classCount[vote] += 1 #suma uno al valor de la key
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #Ordena de mayor a menor
    return sortedClassCount[0][0] #retorna el primer elemento, es decir, la clase mayoritaria

def createTree(dataSet,labels): #Esta funcion se encarga de la creación del arbol de manera recursiva
    classList = [example[-1] for example in dataSet] #La lista de clases pertenece a la ultima columna del dataset, por eso la asigna a classList
    if classList.count(classList[0]) == len(classList):  #Comprueba si el numero de clases es equivalente al tamaño de la lista de clases
        return classList[0]#En caso de que si, significa que todas las clases son iguales por lo que ahi termina la recursividad y retorna la clase
    if len(dataSet[0]) == 1: #De la misma manera, si no hay más caracteristicas en el dataset, deja de dividirse
        return majorityCnt(classList) #Retorna el valor de la clase mayoritaria
    bestFeat = chooseBestFeatureToSplit(dataSet) #Obtiene la mejor caracteristica para dividir
    bestFeatLabel = labels[bestFeat] #Obtiene las etiquetas de aquellos con la mejor caracteristica
    myTree = {bestFeatLabel:{}} #Crea un diccionario de diccionarios cuya llave es la de la etiqueta con la mejor caracteristica
    del(labels[bestFeat]) #elimina las etiquetas de las mejores caracteristicas del vector de etiquetas, en resumen, dividió las etiquetas en 2
    featValues = [example[bestFeat] for example in dataSet] #Obtiene los valores de la mejor caracteristica para despues poder hacer las siguientes divisiones
    uniqueVals = set(featValues) #aqui obtiene los valores únicos de esta caractetistica
    for value in uniqueVals: #for que recorre los valores unicos de la mejor caracteristica
        subLabels = labels[:]       #copia todas las etiqutas, pues si enviara tal cual el valor de labels, entre los subarboles estarian accediendo a la misma variable
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) #asigna por cada uno de los valores únicos de la mejor etiqueta el siguiente subarbol
    return myTree #retorna el arbol                            
    
def classify(inputTree,featLabels,testVec): #Funcion que recorre un arbol para dar el resultado de la clasificacion, tambien es una funcion recursiva
    firstStr = list(inputTree)[0] #como el arbol es un diccionario de diccionarios, con esto obtiene el valor de la primer llave, es decir, el valor de la etiqueta
    secondDict = inputTree[firstStr] #Aqui obtiene el valor de la primera llave, que viene a ser un conjunto de arboles
    featIndex = featLabels.index(firstStr) #Dentro del vector de caracteristicas, obtiene aquella que corresponde con la del nivel actual del arbol
    key = testVec[featIndex] #Obtiene el valor de la caracteristica que corresponde con la del arbol del vector de entrada
    valueOfFeat = secondDict[key] #Se adentra al arbol, obteniendo el siguiente arbol que corresponde a la caracteristica dada por el vector de entrada
    if isinstance(valueOfFeat, dict): #Comprueba si realmente existe ese subarbol
        classLabel = classify(valueOfFeat, featLabels, testVec) #En caso de que si, se llama a si mismo con los nuevos valores del arbol (recursividad)
    else: classLabel = valueOfFeat #en caso contrario, el valor de la etiqueta de clase corresponde al valor de la caracteristica
    return classLabel #retorna la etiqueta de clase

def storeTree(inputTree,filename): #Funcion que guarda un arbol en un archivo
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename): #Funcion que carga un arbol de un archivo
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
    
