import pandas as pd
import matplotlib.pyplot as plt

def plot_grafico(label,lista_metricas,lista_valores,ax,color):
    lista_tuplos = zip(lista_metricas,lista_valores)
    ax.bar(lista_metricas, lista_valores, color=color, label=label,align='center')
    for metric, value in lista_tuplos:
        ax.text(metric , value, f'{value:.2f}', ha='center', va='bottom', color='black',rotation='vertical')


def grafico_seq(df,ax):
    avg_seq = df.mean()['seq']
    max_seq = df.max()['seq']
    min_seq = df.min()['seq']
    med_seq = df.median()['seq']
    plot_grafico('Sequential values',['Avg S', 'Max S', 'Min S', 'Med S'],[avg_seq,max_seq,min_seq,med_seq],ax,'red')

def grafico_par(df,ax):
    avg_par = df.mean()['par']
    max_par = df.max()['par']
    min_par = df.min()['par']
    med_par = df.median()['par']
    plot_grafico('Parallel Open-MP values',['Avg O', 'Max O', 'Min O','Med O'],[avg_par,max_par,min_par,med_par],ax,'lime')

def grafico_cuda(df,ax):
    avg_cuda = df.mean()['cuda']
    max_cuda = df.max()['cuda']
    min_cuda = df.min()['cuda']
    med_cuda = df.median()['cuda']
    plot_grafico('Parallel Cuda values',['Avg C1', 'Max C1', 'Min C1','Med C1'],[avg_cuda,max_cuda,min_cuda,med_cuda],ax,'blue')
 
def grafico_cuda_opt(df,ax):
    avg_cuda = df.mean()['cuda_opt']
    max_cuda = df.max()['cuda_opt']
    min_cuda = df.min()['cuda_opt']
    med_cuda = df.median()['cuda_opt']
    plot_grafico('Parallel Cuda Opt values',['Avg C2', 'Max C2', 'Min C2','Med C2'],[avg_cuda,max_cuda,min_cuda,med_cuda],ax,'orange')
 

def cria_graficos(df,version,ax): 
    grafico_seq(df,ax)
    grafico_par(df,ax)
    grafico_cuda(df,ax)
    grafico_cuda_opt(df,ax)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'N={version}')
    ax.legend()

def x2_aux(df1,num1,df2,num2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 9))  # 2x2 grid for 4 subplots
    axs = axs.flatten()
    cria_graficos(df1,num1,axs[0])
    cria_graficos(df2,num2,axs[1])
    for ax in axs:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate x-axis labels
    plt.tight_layout()
    plt.show()

def x2(df2500,df5000,df10000,df20000):
    x2_aux(df2500,2500,df5000,5000)
    x2_aux(df10000,10000,df20000,20000)



def x4(df2500,df5000,df10000,df20000):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid for 4 subplots
    axs = axs.flatten()
    cria_graficos(df2500,2500,axs[0])
    cria_graficos(df5000,5000,axs[1])
    cria_graficos(df10000,10000,axs[2])
    cria_graficos(df20000,20000,axs[3])
    for ax in axs:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotate x-axis labels
    plt.tight_layout()
    plt.show()

df2500  = pd.read_csv('timing_results_2500.csv' ,sep=';')
df5000  = pd.read_csv('timing_results_5000.csv' ,sep=';')
df10000 = pd.read_csv('timing_results_10000.csv',sep=';')
df20000 = pd.read_csv('timing_results_20000.csv',sep=';')

x2(df2500,df5000,df10000,df20000)

