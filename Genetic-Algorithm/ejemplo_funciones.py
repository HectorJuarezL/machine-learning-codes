import pandas as pd
import numpy as np

data = pd.read_csv('QtyT40I10D100K.csv')

print(data)

print(f' descripcion de transacciones {data["Trans"].describe()}')

trans=data["Trans"].values.astype('>i2')
print(f'\nMuestra de transacciones: {trans}')

print(f'\nValor máximo de la transaccion: {np.max(trans)}')

size=trans.shape[0]
print(f'\nNumero total de trabsacciones: {size}')

print('\nPrimero se convierten los enteros a arreglos binarios')
binary=np.unpackbits(trans.view(np.uint8),axis=0) 
print(f'>Forma del vector de transacciones binario {binary.shape}')
print('\nLa funcion utilizada entrega un vector de 16 veces el tamaño original ya \
      que los datos eran enteros de 16 bits, pero para manejar los datos es más \
      conveniente una matriz por lo que se cambia la forma con reshape')
binary=binary.reshape([size,16,]) 
print(f'\nForma del vector de transacciones binario {binary.shape}')

print('\na continuación se muestra el valor de b[0] el cual corresponde al entero 36 en su forma binaria')
print(binary[0])

print('\na continuación se muestra el valor de b[-1], es decir, el ultimo elemento del arreglo el cual corresponde al entero 980 en su forma binaria')
print(binary[-1])

print('\nahora se obtiene la suma por cada tupla con la funcion sum')
trans_sum=np.sum(binary,axis=1)
print('\nse muestra como ejemplo la suma de la ultima tupla:')
print(trans_sum[-1])

print('\nse eliminan aquellos elementos cuya suma sea menor a 3, ademas\
ya que el valor entero máximo en las transacciones es de 999, \
esto quiere decir que maximo se ocupan 10 bits para representar las transacciones \
por lo que hay 10 productos distintos, entonces los primeros 6 bits no se ocupan.')
d=binary[trans_sum>2,6:]
print(d.shape)

print(f'\ntuplas eliminadas por no tener mas de 2 transacciones: {trans_sum.shape[0]-d.shape[0]}')

def get_soporte(data,items):
    aux=np.zeros([data.shape[0],len(items)],dtype=np.uint8)
    li=len(items)
    if li==0:
        return 0
    for c,i in enumerate(items):
        aux[:,c]=data[:,i]
    return np.count_nonzero(np.sum(aux,axis=1)==li)/data.shape[0]
    

#funcion para obtener la aptitud de una regla de asociación.
#a,b y c son hiperparámetros que permite ponderar los elementos que componen la funcion,
# es decir, el soporte, la confianza y el interes
def get_aptitud(data,x,y,a=1,b=1,c=1):
    if type(x)!=set:
        x={x}
    if type(y)!=set:
        y={y}
    #al evaluar la interseccion de conjuntos, se evitan los casos donde x->x,
    #es decir, se busca que tengan valores distintos x e y
    if len(x.intersection(y))==0:
        supx=get_soporte(data,x)
        supy=get_soporte(data,y)
        supxy=get_soporte(data,x.union(y))
        if supx==0:
            return -1
        elif supy==0:
            return -1
        elif supxy==0:
            return -1
        else:
            confianza=supxy/supx
            interes=supxy/supx/supy
            return a*supx+b*confianza+c*abs(1-interes)
    else:
        return -1

poblacion_size=10
poblacion=np.random.randint(0,3,[poblacion_size,10],dtype=np.int8)-1
print('\nLa poblacion se representa con matrices donde las tuplas representana  los individuos \
y las columnas a cada uno de los items, el valor de -1 indica un antecedente en la regla \
y el 1 un consecuente en la regla. Los ceros son ausencia de los items en la regla\n')
print(poblacion)

def bin2set(a):
    x=[]
    y=[]
    for i,v in enumerate(a):
        if v == -1:
            x.append(i)
        elif v == 1:
            y.append(i)
    return set(x),set(y)

print('\npara poder visualizar los datos en forma de conjuntos se utiliza una funcion propia que se encarga de\
traducir el vector a sus respectivos conjuntos x->y')
x,y=bin2set(poblacion[0])

print(x)

print(y)

print('\nse creó una funcion de ruleta, recibe un vector de probabilidades y su resultado es el indice seleccionado simulando una ruleta')
def ruleta(s):
    prob=0
    s=s/np.sum(s)
    r=np.random.uniform()
    for i in range(0,len(s)):
        prob+=s[i]
        if r<prob:
            return i
    return -1
        

print('\nde igual forma, se creo una funcion para ordenar una lista utilizando una ruleta y un vector de probabilidades\
los valores con mayor probabilidad deberian de ser los primeros de la lista, pero al usar la ruleta le brinda\
la aleatorieada')
def perm_ruleta(prob):
    mapeo=np.arange(len(prob))
    new_lista=np.zeros([len(prob),],dtype=np.uint8)
    p=prob.copy()
    for i in range(len(prob)):
        aux=ruleta(p)
        new_lista[i]=mapeo[aux]
        p=np.delete(p,aux)
        mapeo=np.delete(mapeo,aux)
    return new_lista

print('\nFinalmente está la funcion de epoca donde se realizan los 4 pasos esenciales de un algoritmo genetico.\
\nla seleccion, recombinacion, mutacion y supervivencia selectiva')
def epoca(data,poblacion,a=1,b=1,c=1,mutacion=0.01):
    size,items=poblacion.shape
    aptitud=np.zeros([size,])
    #primero se obtiene la aptitud de cada uno de los individuos
    for i,v in enumerate(poblacion):
        x,y=bin2set(v)
        aptitud[i]=get_aptitud(data,x,y,a,b,c)
        
    #en la variable prioridad se seleccionan en orden dando preferencia a aquellos más aptos
    prioridad=perm_ruleta(aptitud)
    descendientes = np.zeros(poblacion.shape,dtype=np.int8)
    
    #se crean descendientes (recombinacion)
    for i in range(size):
        if i%2==0: #se recombinan los padres de 2 en dos
            r = np.random.choice([True, False], size) #variable auxiliar para eleccion de genes
            descendientes[i]=np.where(r, poblacion[prioridad[i]], poblacion[prioridad[i+1]])
            descendientes[i+1]=np.where(r, poblacion[prioridad[i+1]], poblacion[prioridad[i]])
            #a continuación se realiza la mutacion
            if np.random.random()<mutacion:
                m=np.random.randint(items)
                descendientes[i,m]=np.random.randint(3)-1
            if np.random.random()<mutacion:
                m=np.random.randint(items)
                descendientes[i+1,m]=np.random.randint(3)-1
    
    #se añaden los descendientes a la poblacion
    poblacion = np.concatenate((poblacion,descendientes))
    
    #se calcula de la aptitud de los individuos
    aptitud=np.zeros([size*2,])
    for i,v in enumerate(poblacion):
        x,y=bin2set(v)
        aptitud[i]=get_aptitud(data,x,y,a,b,c)
    #se seleccionan aquellos que sobrevivan, al usar la ruleta no se seleccionan solo los más aptos
    # si no los que tengan la "suerte" de sobrevivir.
    prioridad=perm_ruleta(aptitud)
    return poblacion[prioridad[:size]]

a=.1 #peso que tiene el soporte
b=.1 #peso que tiene la confianza
c=10 #peso que tiene el interes
s=epoca(d,poblacion,a,b,c)

print('\nA continucion se muestra el resultado de aplicar por una epoca/generacion el algoritmo.')
print(s)

x,y=bin2set(s[-1])

print(f'\nla interpretacion del ultimo individuo de la poblacion es la siguiente: {x}->{y}')