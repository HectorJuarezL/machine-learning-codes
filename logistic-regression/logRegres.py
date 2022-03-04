from numpy import *

def loadDataSet(): #Funcion que se encarga de la carga un archivo de datos separados por tabulaciones, es importante recalcar que desde este punto se rellena el valor de x0 con 1
    dataMat = []; labelMat = [] #Variables para guardar la matriz de datos y el vector de etiquetas
    fr = open('testSet.txt') #Abre el archivo
    for line in fr.readlines(): #For que recorre cada una de las lineas del archivo
        lineArr = line.strip().split() #Aqui separa las lineas 
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #añade el valor de x0=1 y los valores leidos del archivo
        labelMat.append(int(lineArr[2])) #Funcion que se encarga de añadir las etiquetas correspondientes a la ultima columna del archivo
    return dataMat,labelMat #retorna la variable con los datos y la variable con las etiquetas

def sigmoid(inX): #Funcion sigmoide
    return 1.0/(1+exp(-inX)) #Retorna un valor entre 0.5 y 1 para inX>=0 y retorna entre 0 y 0.5 para inX<0

def gradAscent(dataMatIn, classLabels): #Funcion núcleo del algoritmo
    dataMatrix = mat(dataMatIn) #Convierte el arreglo bidimensional en una matriz Numpy
    labelMat = mat(classLabels).transpose() #Convierte el arreglo bidimensional en una matriz Numpy
    m,n = shape(dataMatrix) #Obtiene el tamaño de la matriz, m=filas y n=columnas
    alpha = 0.001 #Establece el valor de alpha que es la proporcion por la cual se ma a mover en el ascenso, es decir, un valor mas pequeño requiere de más iteraciones pero da una mejor exatitud al llegar al valor máximo de la funcion
    maxCycles = 500 #Establece el numero máximo de ciclos, entre más ciclos el error deberia de disminuir, pero tambien se puede obtener un sobreaprendizaje (sin mencionar que tomaria mas tiempo)
    weights = random.random(size=(n,1)) #Pesos, estos comienzan con un valor de 1
    for k in range(maxCycles): #Esta es la parte del entrenamiento, el cual se repetira según el numero máximo de ciclos dados
        h = sigmoid(dataMatrix*weights) #Aqui se realiza el producto entre la matriz de datos y los pesos y se obtiene el valor h (correspondientte a la predicción), ya que la matriz es de mxn y los pesos de nx1, h es de mx1
        error = (labelMat - h) #Calcula el error entre el resultado obtenido y el deseado, desde otro punto de vista puede verse como el diferencial de z
        weights = weights + alpha * dataMatrix.transpose()* error #actualiza el valor de la matriz de pesos moviendose en direccion al gradiente  multiplicado por alpha
    return weights #retorna la matriz de pesos

def plotBestFit(weights): #Funcion que se encarga de imprimir el dataset y marcar la linea de separacion dada por los pesos, es interesante el como obtiene esta linea pues toma el primer peso w0 como el bias, w1 la multiplica por el vector x y w2 se encarga de la pendiente.
    import matplotlib.pyplot as plt #importa la libreria para graficar
    dataMat,labelMat=loadDataSet() #carga el dataset
    dataArr = array(dataMat) #convierte el dataset en un arreglo de numpy
    n = shape(dataArr)[0] #obtiene el numero de columnas
    xcord1 = []; ycord1 = [] #arreglos donde se guardaran los vectores de la etiqueta 1
    xcord2 = []; ycord2 = [] #arreglos donde se guardaran los vectores de la etiqueta 2
    for i in range(n): #for para asignar los vectores a su arreglo correspondiente segun la etiqueta
        if int(labelMat[i])== 1: #comprueba si pertenece a la clase 1
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2]) #añade el vector al arreglo de clase 1
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2]) #añade el vector al arreglo de clase 2
    fig = plt.figure() #crea una figura de matplotlib
    ax = fig.add_subplot(111) #acrea un axis a la figura que se usará para graficar los puntos
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') #grafica los puntos de la clase uno de color rojo con forma de cuadrado
    ax.scatter(xcord2, ycord2, s=30, c='green') #grafica los puntos de la clase 2
    x = arange(-3.0, 3.0, 0.1) #arreglo que rellena con elementos desde -3 hasta 3 con un step de 0.1
    y = (-weights[0]-weights[1]*x)/weights[2] #asigna el valor de y dados los pesos y el vector x (esto es lo interesante que mencioné al principio de la funcion)
    ax.plot(x, y) #grafica una linea dadas las coordenadas x,y
    plt.xlabel('X1'); plt.ylabel('X2'); #añade etiquetas a la grafica
    plt.show() #muestra la gráfica

def stocGradAscent0(dataMatrix, classLabels): #Esta variación permite el entrenamiento de conjuntos de datos más grandes además de un aprendizaje "online" lo que significa que puede seguir aprendiendo con nuevos datos 
    m,n = shape(dataMatrix) #obtiene las m filas y n columnas de la matriz
    alpha = 0.01 #establece el valor de alfa
    weights = random.random(size=n)   #inicialzia los pesos en unos
    for i in range(m): #for que recorre las filas del dataset
        h = sigmoid(sum(dataMatrix[i]*weights)) #calcula el valor de h (correspondiente a la prediccion de clase), en este caso es un escalar
        error = classLabels[i] - h #calcula el error entre el valor predecido y el real, siendo el diferencial de xi
        weights = weights + alpha * error * dataMatrix[i] #de igual manera a la anterior funcion de ascenso de gradiente, se mueve en dirección al diferencial de xi
    return weights #retorna la matriz de pesos

def stocGradAscent1(dataMatrix, classLabels, numIter=150): #variación de la funcion anterior, en esta utiliza un alpha variable y un orden aleatorio de los patrones de aprendizaje
    m,n = shape(dataMatrix) #Obtiene las filas y columnas de la matriz
    weights = random.random(size=n)   #Inicialzia los pesos con unos
    for j in range(numIter): #For para recorrer el numero de ciclos dados como parametro
        dataIndex = list(range(m)) #crea una lista de tamaño m con los numeros de 0-m, sirve para saber los indices que ya se usaron
        for i in range(m): #for para recorrer los m elementos
            alpha = 4/(1.0+j+i)+0.0001 #El valor de alpha disminuye con cada iteración, esta nunca llega a 0 por la constante que se le suma
            randIndex = int(random.uniform(0,len(dataIndex))) #obtiene un indice aleatorio entre el numero de elementos que aun hay en dataIndex
            h = sigmoid(sum(dataMatrix[randIndex]*weights)) #esta y las siguientes 2 lineas son igual que en stocGradAscent0 con la diferencia de que usa el índice aleatorio
            error = classLabels[randIndex] - h #calcula el error que viene a ser el diferencial de xi
            weights = weights + alpha * error * dataMatrix[randIndex] #mueve los pesos en direccion al diferencial de xi
            del(dataIndex[randIndex]) #elimina el indice utilizado
    return weights #retorna los pesos

def classifyVector(inX, weights): #Funcion que se encarga de clasificar un vector de entrada inX dados un vector de pesos
    prob = sigmoid(sum(inX*weights)) #simplemente aplica lo ya visto en la teoria, la suma del producto entre el vector de entrada y los pesos y al resultado le aplica la funcion sigmoide
    if prob > 0.5: return 1.0 #si es mayor a 0.5 retorna 1, es decir, que pertenece a la clase 1
    else: return 0.0 #de lo contrario retorna 0, es decir, que pertenece a la clase 0

def colicTest(): #ejemplo para estimar si un caballo con cólicos morirá
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt') #apertura de archivos para la carga de los datos, un archivo contiene los datos de prueba y otro para el entrenamiento
    trainingSet = []; trainingLabels = [] #ya que el autor no tiene una funcion especial para la carga de estos archivos, las siguientes lineas de código se encargan de esto mismo
    for line in frTrain.readlines(): #For que recorre el archivo de datos de entrenamiento
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000) #en esta linea realiza el entrenamiento con 1000 iteraciones y obtiene los pesos de entrenamiento
    errorCount = 0; numTestVec = 0.0 #inicializacion de variables para el conteo de errores y el numero de vectores de prueba
    for line in frTest.readlines(): #For que recorre el archivo de datos de prueba, al mismo tiempo que lee las filas, realiza la clasificacion y la compara con el valor de la etiqueta real
        numTestVec += 1.0 #añade 1 al numero de vectores de prueba
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]): #aqui realiza la clasificación y la compara con el valor de la etiqueta real
            errorCount += 1 #en caso de que no coincidan, añade 1 al conteo de errores
    errorRate = (float(errorCount)/numTestVec) #Calcula el porcentaje de error
    print ("the error rate of this test is: %f" % errorRate) #Imprime el porcentaje de error
    return errorRate #retorna el porcentaje de error

def multiTest(): #Funcion para realizar el test de la funcion anterior varias veces y calcular el error promedio
    numTests = 10; errorSum=0.0 #inicializa el numero de pruebas y el valor de suma de errores
    for k in range(numTests): #for que recorre el numero de tests a realizar
        errorSum += colicTest() #suma acumulativa de error
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))) #imprime el error que corresponde al error acumulado entre el numero de tests
        