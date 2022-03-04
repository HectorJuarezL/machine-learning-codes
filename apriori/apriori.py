from numpy import *

def loadDataSet(): #retorna un pequeño dataset de prueba
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet): #Obtiene todos los items únicos de un dataset
    C1 = [] #Se crea la variable vacia
    for transaction in dataSet: #For que recorre todas las filas del dataset, es decir, todas las transacciones
        for item in transaction: #Recorre cada uno de los objetos de la transacción
            if not [item] in C1: #Comprueba si el objeto no está ya incluido en la variable C1
                C1.append([item]) #En caso de que no está, lo agrega
                
    C1.sort() #Ordena el arreglo
    return list(map(frozenset, C1))#El usar el frozenset hace inmutable la lista, además de que permite usarlo como llave para un diccionario

def scanD(D, Ck, minSupport): #Esta funcion se encarga de obtener el soporte de los conjuntos, recibe de parametro las transacciones del dataset y los items del dataset, además del minSupport que es el margen que nosotros tomaremos para aceptar o no a un soporte e ir "podando" el arbol
    ssCnt = {} #Crea un diccionario vacio donde se guardarán los contadores de ocurrencias
    for tid in D: #Recorre los elementos en D, los cuales son las transacciones del dataset
        for can in Ck: #Recorre los elementos en Ck, los cuales son los items del dataset
            if can.issubset(tid): #Comprueba si el item can está contenido en tid
                if can not in ssCnt: ssCnt[can]=1 #Si el key del item no está en ssCnt lo crea y lo asigna en 1
                else: ssCnt[can] += 1 #En caso de que ya exista, aumenta en 1 su contador
    numItems = float(len(D)) #guarda el tamaño de la variable D en numItems
    retList = [] #Crea un arreglo vacio
    supportData = {} #Crea un diccionario vacio
    for key in ssCnt: #For que recorre todas las llaves (items) en ssCnt
        support = ssCnt[key]/numItems #Calcula el valor del soporte, que es equivalente a el numero de ocurrencias que tiene un item entre el número de items
        if support >= minSupport: #Comprueba si cumple con el valor minimo requerido de soporte
            retList.insert(0,key) #En caso de que si, lo guarda al inicio de la lista retList
        supportData[key] = support #En el diccionario de supportData se guarda el valor de soporte para todas las relaciones
    return retList, supportData #Retorna la lista de los soportes que superaron el minimo además de otro diccionario con el valor de los soportes de todas los items.

def aprioriGen(Lk, k): #Permite crear los elementos del conjunto potencia con k items unidos.
    retList = []
    lenLk = len(Lk) #Calcula el tamaño de Lk
    for i in range(lenLk): #for para recorrer todos los elementos de Lk
        for j in range(i+1, lenLk):  #De manera análoga a las fichas de dominó sin contar las mulas, todas las combinaciones posteriores no necesitan de las anteriores, por esto comienza con i+1 y termina en la longitud de Lk
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] #Obtiene los k-2 elementos de LKi y Lkj y los guarda en L1 y L2 respectivamente
            L1.sort(); L2.sort() #Ordena las listas
            if L1==L2: #Comprueba si los primeros k-2 elementos son iguales
                retList.append(Lk[i] | Lk[j]) #Agrega la union de los conjuntos
    return retList #Retorna la lista

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet) #Obtiene los items únicos del dataset
    D = list(map(set, dataSet)) #Convierte a conjuntos las filas del dataset, es decir, cada elemento de D es una transacción.
    L1, supportData = scanD(D, C1, minSupport) #Obtiene los valores de soporte
    L = [L1] #Crea una variable que contiene a L1 como un elemento de un arreglo, es decir, se guardan los soportes de N+2 relaciones pues el primer elemento que es el cero, tiene 2 relaciones
    k = 2  #Como se comienza con 2 relaciones, k es igual a 2
    while (len(L[k-2]) > 0): #Comprueba que haya elementos en la Lk a revisar, es decir, cuando la Lk recibida no tenga elementos, termina el while
        Ck = aprioriGen(L[k-2], k) #
        Lk, supK = scanD(D, Ck, minSupport)#Obtiene una nueva Lk y su correspondiente soporte
        supportData.update(supK) #como lo dice en el método, actualiza los valores del diccionario supportData con los recibidos de supK
        L.append(Lk) #Agrega Lk a la lista L para continuar con la siguiente iteración del while
        k += 1 #Agrega 1 al valor de k para continuar con la siguiente iteracion del while
    return L, supportData #Retorna L la cual es una variable con todos los soportes 

def generateRules(L, supportData, minConf=0.7):  #Obtiene las reglas de asociacion entre los elementos a partir de los resultados de supportData
    bigRuleList = []
    for i in range(1, len(L)):#Solo obtiene los conjuntos con 2 o mas items
        for freqSet in L[i]: #Obtiene cada uno de los conjuntos de L[i]
            H1 = [frozenset([item]) for item in freqSet] #Obtiene cada uno de los elementos del conjunto freqSet
            if (i > 1): #Con un solo elemento no hay relación por lo que pasa al else
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf) #Crea una regla a partir de los datos del conjunto
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf) #Crea la configuración inicial para cada uno de los conjuntos de L
    return bigRuleList         #Retorna la lista de reglas

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #Crea la variable que retornará como resultado
    for conseq in H: #En este for recorre todos los conjuntos que se encuentran en H
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calcula la confiabilidad, es decir, que tanta reelevancia tiene el dato de soporte
        if conf >= minConf: #comprueba si supera el minimo dado como parámetro
            print(freqSet-conseq,'-->',conseq,'conf:',conf) #Imprime el resultado de confiabilidad
            brl.append((freqSet-conseq, conseq, conf)) #Agrega el valor de conseq y conf a la lista recibida
            prunedH.append(conseq) #de igual manera, es capaz de retornar una lista propia con lo que agrega aqui los valores de conseq
    return prunedH #retorna la lista propia de solo los valores de conseq

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0]) 
    if (len(freqSet) > (m + 1)): #Busca si hay más combinaciones, es decir, se adentra en el arbol de expansion
        Hmp1 = aprioriGen(H, m+1)#Crea los siguientes candidatos, es decir, las combinaciones siguientes
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf) #Calcula la confidencialidad de estos
        if (len(Hmp1) > 1):    #Para poder establecer una nueva regla necesita que haya almenos 2 conjuntos
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf) #De manera recursiva, se vuelve a llamar a si misma la funcion hasta que ya no haya 2 conjuntos disponibles
            
def pntRules(ruleList, itemMeaning): #Muestra las reglas relacionando la lista de reglas con el significado de los objetos, para esto se necesita otra variable que mapee los items en numero con su significado.
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print("     ")#print a blank line
        
            
