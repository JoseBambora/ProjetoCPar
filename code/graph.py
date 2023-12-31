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

def fun_speed_up(avg_seq,avg):
    return [e[0] / e[1] for e in zip(avg_seq,avg)]

def plot_points_comparison_and_speed_up(df2500, df5000, df10000, df20000):
    ordem = [2500, 5000, 10000, 20000]

    avg_seq      = [ df2500.mean()['seq']     , df5000.mean()['seq']     , df10000.mean()['seq']     , df20000.mean()['seq']     ]
    avg_par      = [ df2500.mean()['par']     , df5000.mean()['par']     , df10000.mean()['par']     , df20000.mean()['par']     ]
    avg_cuda     = [ df2500.mean()['cuda']    , df5000.mean()['cuda']    , df10000.mean()['cuda']    , df20000.mean()['cuda']    ]
    avg_cuda_opt = [ df2500.mean()['cuda_opt'], df5000.mean()['cuda_opt'], df10000.mean()['cuda_opt'], df20000.mean()['cuda_opt']]

    avg_par_speed_up      = fun_speed_up(avg_seq, avg_par)
    avg_cuda_speed_up     = fun_speed_up(avg_seq, avg_cuda)
    avg_cuda_opt_speed_up = fun_speed_up(avg_seq, avg_cuda_opt)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(ordem, avg_seq, linestyle='--', marker='o', color='red', label="Sequential")
    plt.plot(ordem, avg_par, linestyle='--', marker='o', color='lime', label="Open MP")
    plt.plot(ordem, avg_cuda, linestyle='--', marker='o', color='blue', label="Cuda v1")
    plt.plot(ordem, avg_cuda_opt, linestyle='--', marker='o', color='orange', label="Cuda v2")

    plt.xticks(ordem)
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance comparison - Time')

    plt.subplot(1, 2, 2)
    plt.plot(ordem, avg_par_speed_up, linestyle='--', marker='o', color='lime', label="Open MP")
    plt.plot(ordem, avg_cuda_speed_up, linestyle='--', marker='o', color='blue', label="Cuda v1")
    plt.plot(ordem, avg_cuda_opt_speed_up, linestyle='--', marker='o', color='orange', label="Cuda v2")

    plt.xticks(ordem)
    plt.xlabel('N')
    plt.ylabel('Speed up')
    plt.title('Performance comparison - Speed up')

    plt.tight_layout()
    plt.show()

df2500  = pd.read_csv('timing_results_2500.csv' ,sep=';')
df5000  = pd.read_csv('timing_results_5000.csv' ,sep=';')
df10000 = pd.read_csv('timing_results_10000.csv',sep=';')
df20000 = pd.read_csv('timing_results_20000.csv',sep=';')
x2(df2500, df5000, df10000, df20000)
plot_points_comparison_and_speed_up(df2500, df5000, df10000, df20000)

# plot_points_speed_up(df2500,df5000,df10000,df20000)


