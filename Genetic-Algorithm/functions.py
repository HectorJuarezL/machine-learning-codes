import pandas as pd
import numpy as np



def get_soporte(data,items):
    aux=np.zeros([data.shape[0],len(items)],dtype=np.uint8)
    li=len(items)
    if li==0:
        return 0
    for c,i in enumerate(items):
        aux[:,c]=data[:,i]
    return np.count_nonzero(np.sum(aux,axis=1)==li)/data.shape[0]
    

def get_aptitud(data,x,y,a,b,c):
    if type(x)!=set:
        x={x}
    if type(y)!=set:
        y={y}
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
            aptitud=a*supx+b*confianza+c*abs(1-interes)
            return aptitud
    else:
        return -1


def get_params(data,x,y,a,b,c):
    if type(x)!=set:
        x={x}
    if type(y)!=set:
        y={y}
    if len(x.intersection(y))==0:
        supx=get_soporte(data,x)
        supy=get_soporte(data,y)
        supxy=get_soporte(data,x.union(y))
        if supx==0:
            return [-1]
        elif supy==0:
            return [-1]
        elif supxy==0:
            return [-1]
        else:
            confianza=supxy/supx
            interes=supxy/supx/supy
            aptitud=a*supx+b*confianza+c*abs(1-interes)
            return aptitud,supx,supy,confianza,interes
    else:
        return [-1]

def bin2set(a):
    x=[]
    y=[]
    for i,v in enumerate(a):
        if v == -1:
            x.append(i)
        elif v == 1:
            y.append(i)
    return set(x),set(y)

def perm_ruleta(prob):
    mapeo=np.arange(len(prob))
    new_lista=np.zeros([len(prob),],dtype=np.uint8)
    p=prob.copy()
    for i in range(len(prob)):
        aux=ruleta(p)
        new_lista[i]=mapeo[aux]
        p=np.delete(p,aux)
        #p=p/np.sum(p)
        mapeo=np.delete(mapeo,aux)
    return new_lista

def ruleta(s,):
    prob=0
    s=s/np.sum(s)
    r=np.random.uniform()
    for i in range(0,len(s)):
        prob+=s[i]
        if r<prob:
            return i
    return -1

def epoca(data,poblacion,a=1,b=1,c=1,mutacion=0.01):
    size,items=poblacion.shape
    aptitud=np.zeros([size,])
    for i,v in enumerate(poblacion):
        x,y=bin2set(v)
        aptitud[i]=get_aptitud(data,x,y,a,b,c)
    prioridad=perm_ruleta(aptitud)
    descendientes = np.zeros(poblacion.shape,dtype=np.int8)
    
    for i in range(size):
        if i%2==0:
            r = np.random.choice([True, False], items)
            descendientes[i]=np.where(r, poblacion[prioridad[i]], poblacion[prioridad[i+1]])
            descendientes[i+1]=np.where(r, poblacion[prioridad[i+1]], poblacion[prioridad[i]])
            for j in range(items):
                if np.random.random()<mutacion:
                    descendientes[i,j]=np.random.randint(3)-1
                if np.random.random()<mutacion:
                    descendientes[i+1,j]=np.random.randint(3)-1
    poblacion = np.concatenate((poblacion,descendientes))
    aptitud=np.zeros([size*2,])
    for i,v in enumerate(poblacion):
        x,y=bin2set(v)
        aptitud[i]=get_aptitud(data,x,y,a,b,c)
    prioridad=perm_ruleta(aptitud)
    return poblacion[prioridad[:size]]

def process(data,poblacion_inicial,a=0.1,b=0.1,c=10,epocas=50):
    p=poblacion_inicial.copy()
    print(p)
    for i in range(epocas):
        print(f'\n\tGeneración: {i+1}\n')
        p=epoca(data,p,a,b,c)
        for i,v in enumerate(p):
            x,y=bin2set(v)
            params=get_params(data,x,y,a,b,c)
            if params[0]>0:
                print(f'x:{str(x):^18} -> y:{str(y):^18} apt={params[0]:.2f}  supp(x)={params[1]:.2f}  supp(y)={params[2]:.2f}  conf(x,y)={params[3]:.2f}  lift(x,y)={params[4]:.2f}')
            else:
                print(f'x:{str(x):^18} -> y:{str(y):^18} apt={params[0]} ')


def read_file(fname):
    import os.path
    if not os.path.isfile(fname):
        import zipfile
        with zipfile.ZipFile(fname.replace('.csv','.zip'),"r") as zip_ref:
            zip_ref.extractall("./")
    data = pd.read_csv(fname)
    return data                
                   
def run(fname,start=0,size=1000,a=0.1,b=0.1,c=10,p_size=10,seed=1,epocas=50,ne=1):
    """
        start=numero de transaccion de inicio
        size=tamaño de muestra para el algoritmo
        a=peso que tiene el soporte
        b=peso que tiene la confianza
        c=peso que tiene el interes
        p_size=numero de individuos de la poblacion
        seed=semilla para obtencion de numeros pseudoaleatorios
    """
    data = read_file(fname)
    trans=data["Trans"].values.astype('>i2')
    trans=trans[start:start+size]

    binary=np.unpackbits(trans.view(np.uint8),axis=0) 

    binary=binary.reshape([size,16,]) 
    print(f'tamaño de las transacciones: {binary.shape}')
    trans_sum=np.sum(binary,axis=1)


    d=binary[trans_sum>2,6:]
    print(f'tamaño de las transacciones mayor o igual a 3: {d.shape}')
    print(f'tuplas eliminadas (con 2 transacciones o menos): {trans_sum.shape[0]-d.shape[0]}')

    np.random.seed(seed)

    if ne==1:

        print('Experimento 1, comenzando cada individuo con reglas formadas por solo un antecedente y un consecuente')

        poblacion_inicial=np.zeros([p_size,10],dtype=np.int8)
        ant=np.random.randint(0,10,[p_size,1],dtype=np.int8)
        con=np.random.randint(1,10,[p_size,1],dtype=np.int8)
        for i in range(p_size):
            poblacion_inicial[i,ant[i]]=-1
            consecuente=(ant[i]+con[i])%10
            poblacion_inicial[i,consecuente]=1

        process(d,poblacion_inicial,a,b,c,epocas)
    elif ne==2:

        print('Experimento 2, generando de manera totalmente aleatoria las reglas de asociación')
        np.random.seed(seed)
        poblacion_inicial=np.random.randint(0,3,[p_size,10],dtype=np.int8)-1
        process(d,poblacion_inicial,a,b,c,epocas)
    else:
        print('Solo existen 2 experimentos.')

