# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 23:03:46 2021

@author: Alex Almeida
"""
#-------------------------------------------------------------------------    
#1. Importando módulos
#------------------------------------------------------------------------- 

import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg as la

#-------------------------------------------------------------------------    
#2. Declarando as funções
#------------------------------------------------------------------------- 

def importa_dados (xlsx): 
    # Recebe uma string do tipo 'nome_do_arquivo.xlsx'
    # Importa duas abas do arquivo .xls nomeadas como "nos" e "barras"
    # Retorna o conteúdo das abas em dois conjuntos de dados
    nos    = pd.read_excel(xlsx,'nos')
    barras = pd.read_excel(xlsx,'barras')    
    return nos, barras

def lista_gl (u,v,teta,nn):
    # Recebe as listas dos graus de liberdade u, v e teta e o número de nós (nn)
    # Concatena as listas e agrupa por nós
    # Retorna uma lista com os graus de Liberdade do sistema agrupados em função do seu nó
    a_gls = []
    for i in range (nn):
        gls_i  = [u[i],v[i],teta[i]]
        a_gls.append(gls_i)    
    return a_gls

def matriz_id_nos (noN,noF,nb):
    # Recebe as listas de Nós Iniciais (noN) e de Nós Finais (noF) e o número de barras (nb)
    # Retorna a Matriz de identificação dos nós( Nó inicial (N) e Nó Final (F) de cada barra)
    IDN      = np.zeros((2,nb))
    IDN[0,:] = noN
    IDN[1,:] = noF
    return IDN

def matriz_id_barras (IDN,nb):     
    # Recebe a Matriz id dos nós (IDN) e o número de barras (nb)
    # Retorna a Matriz de identificação das Barras em relação aos Graus de Liberdade
    IDB = np.zeros((6,nb)) #Graus de liberdade por barra, número de barras
    for i in range(3):
        IDB[i,:]   = IDN[0,:]*3-2+i
        IDB[i+3,:] = IDN[1,:]*3-2+i
    return IDB

def geometria (a, b, c, d):
    # Recebe a lista com a base(direção z) e a altura(x/y) das barras do pórtico
    # Retorna a lista com as áreas (A) e as Inércias (I) das barras
    Area = []
    I    = []
    for i in range (len(a)):
        Ai = a[i]*b[i] + c[i]*d[i]
        Area.append(Ai)        
        A1  = a[i]*b[i]
        A2  = c[i]*d[i]
        AT  = A1 + A2
        A1x = b[i]/2        
        A2x = (d[i]/2) + b[i]        
        xg  = (A1*A1x)/AT + (A2 *A2x)/AT        
        Ii = (((a[i]*(b[i]**3))/12)+(A1*((xg-A1x)**2))) + (((c[i]*(d[i]**3))/12)+(A2*((A2x-xg)**2)))
        I.append(Ii)   
    return Area, I

def L_e_coss(IDN,nb,cx,cy):
    # Recebe a Matriz id dos nós (IDN), o número de barras (nb) e as coordenadas Cx e Cy
    # Retorna a lista do Comprimento de cada barra e os cossenos diretores
    Lx   = np.zeros(nb)
    Ly   = np.zeros(nb)
    cosx = np.zeros(nb)
    cosy = np.zeros(nb)
    L    = np.zeros(nb)
    for n in range (nb):
        k1      = int(IDN[0,n] -1)  # Indexador da matriz IDN
        k2      = int(IDN[1,n] -1)  # Indexador da matriz IDN
        Lx[n]   = cx[k2] - cx[k1]
        Ly[n]   = cy[k2] - cy[k1]
        L[n]    = np.sqrt(Lx[n]**2 + Ly[n]**2)
        cosx[n] = Lx[n]/L[n]
        cosy[n] = Ly[n]/L[n]
    return L, cosx, cosy

def matrizes (ngl,nb,L,cosx,cosy,IDB,RHO,E,A,I):
    # Recebe:
    # ngl = Número de graus de liberdade
    # nb  = número de barras
    # IDN = Matriz id do Nó inicial (N) e Nó Final (F) de cada barra
    # IDB = Matriz id das Barras em relação aos Graus de Liberdade
    # cx  = lista com as coordenadas em X dos nós
    # cy  = lista com as coordenadas em Y dos nós
    # E   = Módulo de elasticidade (N/m2)
    # RHO = Massa específica (Kg/m³)
    # A   = Lista com as áreas (m2) das barras
    # I   = Lista com as Inércias (m4) das barras
    # Retorna as Matrizes de Rigidez (K) e de Massa (M)    
    
    # Criando as Matrizes 
    K    = np.zeros((ngl,ngl)) 
    M    = np.zeros((ngl,ngl))
    
    for i in range (nb):
    # Matriz de rigidez local da barra
        k = np.array([[E*A[i]/L[i], 0, 0, -E*A[i]/L[i],0 ,0 ],
                      [0, 12*E*I[i]/(L[i]**3), 6*E*I[i]/(L[i]**2), 0, -12*E*I[i]/(L[i]**3),6*E*I[i]/(L[i]**2)],
                      [0,6*E*I[i]/(L[i]**2), 4*E*I[i]/L[i], 0, -6*E*I[i]/(L[i]**2), 2*E*I[i]/L[i] ],
                      [-E*A[i]/L[i], 0, 0, E*A[i]/L[i],0 ,0 ],
                      [0, -12*E*I[i]/(L[i]**3), -6*E*I[i]/(L[i]**2), 0,12*E*I[i]/(L[i]**3),-6*E*I[i]/(L[i]**2)],
                      [0,6*E*I[i]/(L[i]**2), 2*E*I[i]/L[i], 0,-6*E*I[i]/(L[i]**2), 4*E*I[i]/L[i] ]])

    # Matriz de massa local da barra
        m = ((RHO*A[i]*L[i])/420)*np.array([[140, 0, 0, 70, 0, 0],
                                            [0, 156, 22*L[i], 0, 54, -13*L[i]],
                                            [0, 22*L[i], 4*(L[i]**2), 0, 13*L[i], -3*(L[i]**2)],
                                            [70, 0, 0, 140, 0, 0],
                                            [0, 54, 13*L[i], 0, 156, -22*L[i]],
                                            [0, -13*L[i], -3*(L[i]**2), 0, -22*L[i], 4*(L[i]**2)]])
    
    # Matriz de rotação
        tau = np.array([[cosx[i], cosy[i], 0, 0 ,0 ,0],
                        [-cosy[i], cosx[i],0, 0, 0, 0],
                        [0,0,1,0,0,0],                     
                        [0,0,0,cosx[i], cosy[i], 0],
                        [0, 0, 0,-cosy[i], cosx[i],0],
                        [0,0,0,0,0,1]])

    # Matrizes locais rotacionadas
        k_r = np.dot(np.dot(tau.T, k),tau)
        m_r = np.dot(np.dot(tau.T, m),tau)

    # Alocação das matrizes locais na matriz global
        k_rG = np.zeros((ngl,ngl))
        a1 = int(IDB[0,i]-1)
        a2 = int(IDB[2,i])
        a3 = int(IDB[3,i]-1)
        a4 = int(IDB[5,i])
        k_rG[a1:a2,a1:a2] = k_r[0:3,0:3]
        k_rG[a3:a4,a1:a2] = k_r[3:6,0:3]
        k_rG[a1:a2,a3:a4] = k_r[0:3,3:6]
        k_rG[a3:a4,a3:a4] = k_r[3:6,3:6]
        K += k_rG 

        m_rG = np.zeros((ngl,ngl))
        a1 = int(IDB[0,i]-1)
        a2 = int(IDB[2,i])
        a3 = int(IDB[3,i]-1)
        a4 = int(IDB[5,i])
        m_rG[a1:a2,a1:a2] = m_r[0:3,0:3]
        m_rG[a3:a4,a1:a2] = m_r[3:6,0:3]
        m_rG[a1:a2,a3:a4] = m_r[0:3,3:6]
        m_rG[a3:a4,a3:a4] = m_r[3:6,3:6]
        M += m_rG 
    return K,M

def remover_glr(K,M,u,v,teta,ur,vr,tetar):
    # Recebe:
    # K           = Matriz de rigidez
    # M           = Matriz de massa
    # u,v,teta    = Lista dos graus de liberdade
    # ur,vr,tetar = Lista dos graus de liberdade restringidos    
    # Retorna as Matrizes de Rigidez (Kf) e de Massa (Mf) sem os graus restringidos

    # Montar array com os Graus de Liberdade Restringidos 
    gl         = np.array(u+v+teta)
    id_glr     = np.array(ur+vr+tetar)
    glr        = np.trim_zeros(sorted(gl*id_glr))
    remover_gl = np.array(glr)-1

    # Deletar Linhas e Colunas restringidas das matrizes K e M
    Ki = np.delete(K, remover_gl,axis=0)
    Kf = np.delete(Ki, remover_gl,axis=1)  
    Mi = np.delete(M, remover_gl,axis=0)
    Mf = np.delete(Mi, remover_gl,axis=1)  
    return Kf, Mf

def EIG(Kf,Mf):
    # Recebe:
    # Kf  = Matriz de rigidez restringida
    # Mf  = Matriz de massa restringida
    # Soluciona o problema de Autovalores e Autovetores
    # Retorna as frequencias Naturais wk(rad/s) e fk(Hz)
    # Retorna a matriz com os modos de vibração (Phi)
    lamb,Phi  = sc.eig(Kf,Mf)
    index_eig = lamb.argsort()    # indexando em ordem crescente
    lamb      = lamb[index_eig]   # Aplicando indexador
    Phi       = Phi[:,index_eig]  # Aplicando indexador
    w2        = np.real(lamb)     # Extraindo apenas a parte real de lambda
    wk        = np.sqrt(w2)       # rad/s
    fk        = wk/2/np.pi        # Hz    
    return wk, fk, Phi

def amortecimento(zeta,wk,Mf,Kf):
    # Recebe:
    # zeta = Razão de amortecimento típica para estrutura em análise
    # wk   = Frequencias naturais(rad/s) do primeiro e segundo Modos
    # Kf   = Matriz de rigidez restringida
    # Mf   = Matriz de rigidez restringida
    # Retorna a Matriz de amortecimento de Rayleigh (Cf)    
    a  = zeta*2*wk[0]*wk[1]/(wk[0]+wk[1])  # parâmetro da matriz de massa
    b  = zeta*2/(wk[0]+wk[1])              # parâmetro da matriz de rigidez
    Cf = a*Mf + b*Kf
    return Cf

#-------------------------------------------------------------------------    
#3. Dados de Entrada  
#------------------------------------------------------------------------- 
nos, barras = importa_dados('dados_de_entradaEdificioOriginal.xlsx')

#nos, barras = importa_dados('dados_de_entradaEdificioOriginal.xlsx')
RHO         = 2500                           # Massa específica (Kg/m³)
fck         = 50                             # mPa
E           = 5600*((fck)**(1/2))*0.85*10**6 # Módulo de elasticidade (N/m2)
zeta        = 0.02                           # Razão de amortecimento típica para estrutura aporticada de concreto

#-------------------------------------------------------------------------     
#4. Matrizes auxiliares 
#------------------------------------------------------------------------- 

n_id     = list(nos['Nó'])           # Identificação dos Nós
nn       = len(list(nos['Cx']))      # número de nós
ngl      = len(list(nos['Cx']))*3    # número de graus de liberdade 
ug       = list(nos['u'])[0:nn]      # Identificação do Grau de Liberdade
vg       = list(nos['v'])[0:nn]      # Identificação do Grau de Liberdade
tetag    = list(nos['teta'])[0:nn]   # Identificação do Grau de Liberdade
ur       = list(nos['ur'])[0:nn]     # Identificação dos Graus de Liberdade restringidos
vr       = list(nos['vr'])[0:nn]     # Identificação dos Graus de Liberdade restringidos
tetar    = list(nos['tetar'])[0:nn]  # Identificação dos Graus de Liberdade restringidos
cx       = list(nos['Cx'])[0:nn]     # coordenadas em X dos nós
cy       = list(nos['Cy'])[0:nn]     # coordenadas em Y dos nós
cz       = list(nos['Cz'])[0:nn]     # coordenadas em Z dos nós
noN      = list(barras['noN'])       # Nós Iniciais
noF      = list(barras['noF'])       # Nós Finais
basea    = list(barras['a'])         # Base dos elementos do pórtico (direção z)
alturab  = list(barras['b'])         # Altura dos elementos do pórtico (direção x/y)
basec    = list(barras['c'])         # base segunda parte do pilar em L
alturad  = list(barras['d'])         # Altura segunda parte do pilar em L
nb       = len(noN)                  # Número de Barras
a_gls    = lista_gl(ug,vg,tetag,nn)  # Graus de Liberdade agrupados por nó
IDN      = matriz_id_nos(noN,noF,nb) # Matriz id do Nó inicial (N) e Nó Final (F) de cada barra
IDB      = matriz_id_barras(IDN,nb)  # Matriz id das Barras em relação aos Graus de Liberdade
Area, I  = geometria(basea,alturab,basec,alturad)        # Lista com as áreas (m2) e as Inércias (m4) das barras

# Comprimento de cada barra e cossenos diretores
L, cosx, cosy = L_e_coss(IDN,nb,cx,cy)

#-------------------------------------------------------------------------        
#5. Matrizes de Rigidez e Massa
#-------------------------------------------------------------------------  

#5.1 Definindo as Matrizes de Rigidez (K) e Massa (M) Iniciais do pórtico
K,M   = matrizes(ngl,nb,L,cosx,cosy,IDB,RHO,E,Area,I)

#5.2 Removendo os graus de liberdade restringidos
K,M = remover_glr(K,M,ug,vg,tetag,ur,vr,tetar)

#-------------------------------------------------------------------------     
#6. Matriz de Massa das Lajes (Lumped)
#------------------------------------------------------------------------- 

#6.1 Definindo a Matriz de Massa Lumped
ML = np.zeros((len(M),len(M))) 

nostemp       = list(np.arange(1,(nn+1),1))
nos_externos  = sorted(list(np.arange(4,(nn+4),4)) + list(np.arange(1,(nn+1),4)))
relacao_gl_no = list(np.arange(0,(2*nn),2))  #de zero até o dobro de nós
gl_externos   = []

for i in range(len(nostemp)+1):
    if i in nos_externos:
        utemp = nostemp[i-1]+relacao_gl_no[i-1]
        vtemp = utemp + 1
        ttemp = utemp + 2
        gl_externos.append(utemp)
        gl_externos.append(vtemp)
        gl_externos.append(ttemp)

for i in range(len(ML)):
     for j in range(len(ML)):
        if i == j:
            if i+1 in gl_externos:
                ML[i,j] = 3526.215  #Kg
            else:
                ML[i,j] = 5356.4175 #Kg


#6.2 Somando a Massa das Lajes à Matriz de Massa (Consistente + Lumped)
M  += ML

#-------------------------------------------------------------------------        
#7. Frequências Naturais e Modos de Vibração
#------------------------------------------------------------------------- 
# Frequencias Naturais wk(rad/s) e fk(Hz) e Modos de vibração (Phi)
wk, fk, Phi = EIG(K,M)

#-------------------------------------------------------------------------        
#8. Matriz de Amortecimento
#------------------------------------------------------------------------- 
C = amortecimento(zeta,wk,M,K)

#-------------------------------------------------------------------------        
#9. Propriedades dinâmicas
#------------------------------------------------------------------------- 
Minv = np.linalg.inv(M)

dynsys =  {"M":M,"K":K,"C":C,"Minv":Minv}


#-------------------------------------------------------------------------        
#10. Força do vento
#------------------------------------------------------------------------- 

# Observações:
# Importar para o spyder a Matriz de Força gerada em outro arquivo

F = F0  # matriz com 300s, em dt =0.0005, de Força de arrasto


# Definindo a posição da Força máxima no último pavimento
lista_fa = list(F[408,:])        # lista das forças no último pavimento (gl 408)
fmax     = max(lista_fa)         # valor máximo na lista de forças do ultimo pavimento
fposmax = lista_fa.index(fmax)   # Posição da força máxima no tempo

# Definindo intervalo de análise de 15s com a força máxima no centro
dur     = 15    #seg
dt      = 0.0005 #seg

F = F[:,(fposmax-(int((dur/dt)/2))):(fposmax+(int((dur/dt)/2)))] #novo vetor de forças

#Conferindo se a força máxima está correta
lista_fa2 = list(F[408,:])        # lista das forças no último pavimento (gl 408)
fmax2     = max(lista_fa2)         # valor máximo na lista de forças do ultimo pavimento
fposmax2 = lista_fa2.index(fmax2)   # Posição da força máxima no tempo
assert fmax == fmax2

# Newmark
#------------------------------------------------------------------------- 
#4. Montar arrays Aceleração, Velocidade e Deslocamento
#-------------------------------------------------------------------------       

tf      = int(dur/dt)
t       = np.linspace(0,dur,tf)
n       = len(F[:,0])
Acc     = np.zeros((n,tf))   #Aceleração (m/s²)
v       = np.zeros((n,tf))   #Velocidade (m/s)
U       = np.zeros((n,tf))   #Deslocamento (m) 

#------------------------------------------------------------------------- 
#5. Determinar as constantes do método de Newmark
#-------------------------------------------------------------------------
delta  = 0.5
alfa1  = 0.25
a10    = 1/(alfa1*(dt**2))
a11    = 1/(alfa1*dt)
a12    = (1/(2*alfa1))-1
a13    = delta/(dt*alfa1)
a14    = (delta/alfa1) - 1
a15    = (dt/2)*((delta/alfa1) - 2)
C1     = np.linalg.inv(a10*M + a13*C + K)
Acc[:,0] = np.dot(np.linalg.inv(M),(F[:,0]-np.dot(C,v[:,0])-np.dot(K,U[:,0]))) #aceleração no tempo zero

#------------------------------------------------------------------------- 
#6. Resolver a equação de equilíbrio dinâmico
#-------------------------------------------------------------------------

for i in range(tf-1):

    var1     = F[:,i+1]+np.dot(M,(a10*U[:,i]+ a11*v[:,i] + a12*Acc[:,i]))+np.dot(C,(a13*U[:,i]+ a14*v[:,i] 
                                                                                    + a15*Acc[:,i]))
    U[:,i+1]   = np.dot(C1,var1)
    v[:,i+1]   = a13*(U[:,i+1] - U[:,i]) - a14*v[:,i] - a15*Acc[:,i]    
    Acc[:,i+1] = a10*(U[:,i+1] - U[:,i]) - a11*v[:,i] - a12*Acc[:,i] 