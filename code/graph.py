import pandas as pd
import matplotlib.pyplot as plt

def plot_grafico(label,lista_metricas,lista_valores,ax,color):
    lista_tuplos = zip(lista_metricas,lista_valores)
    ax.bar(lista_metricas, lista_valores, color=color, label=label)
    for metric, value in lista_tuplos:
        ax.text(metric , value, f'{value:.2f}', ha='center', va='bottom', color='black', fontweight='bold')


def grafico_seq(df,ax):
    avg_seq = df.mean()['seq']
    max_seq = df.max()['seq']
    min_seq = df.min()['seq']
    med_seq = df.median()['seq']
    plot_grafico('Sequential values',['Avg Seq', 'Max Seq', 'Min Seq', 'Med Seq'],[avg_seq,max_seq,min_seq,med_seq],ax,'red')

def grafico_par(df,ax):
    avg_par = df.mean()['par']
    max_par = df.max()['par']
    min_par = df.min()['par']
    med_par = df.median()['par']
    plot_grafico('Parallel Open-MP values',['Avg Open-MP', 'Max Open-MP', 'Min Open-MP','Med Open-MP'],[avg_par,max_par,min_par,med_par],ax,'green')

def grafico_cuda(df,ax):
    avg_cuda = df.mean()['cuda']
    max_cuda = df.max()['cuda']
    min_cuda = df.min()['cuda']
    med_cuda = df.median()['cuda']
    plot_grafico('Parallel Cuda values',['Avg Cuda', 'Max Cuda', 'Min Cuda','Med Cuda'],[avg_cuda,max_cuda,min_cuda,med_cuda],ax,'blue')
 
def cria_graficos(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    grafico_seq(df,ax)
    grafico_par(df,ax)
    grafico_cuda(df,ax)
    ax.set_xlabel('Versions')
    ax.set_ylabel('Time (s)')
    ax.set_title('Histogram of Avg, Max and Min Values for Each Version')
    ax.legend()
    plt.tight_layout()
    plt.show()


df2500  = pd.read_csv('timing_results_2500.csv' ,sep=';')
df5000  = pd.read_csv('timing_results_5000.csv' ,sep=';')
df10000 = pd.read_csv('timing_results_10000.csv',sep=';')
# df20000 = pd.read_csv('timing_results_20000.csv',sep=';')
cria_graficos(df2500)
cria_graficos(df5000)
cria_graficos(df10000)

