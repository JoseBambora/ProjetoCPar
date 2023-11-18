import matplotlib.pyplot as plt

def plot_points(points_ganho_task,points_ganho_thread, points_esperado):
    x_values_gtask, y_values_gtask = zip(*points_ganho_task)
    x_values_gthread, y_values_gthread = zip(*points_ganho_thread)
    x_values_e, y_values_e = zip(*points_esperado)  
    plt.plot(x_values_e, y_values_e, linestyle='--', marker='o', color='green', label="Theorical")
    plt.plot(x_values_gtask, y_values_gtask, linestyle='--', marker='o', color='red',   label="Real tasks")
    plt.plot(x_values_gthread, y_values_gthread, linestyle='--', marker='o', color='yellow',   label="Real threads")

    plt.xticks(x_values_gtask)
    plt.xlabel('Nr of Threads')
    plt.ylabel('Gain')
    plt.title('Performance comparison')
    plt.legend()
    plt.show()

x1  = 43.423
x2  = 39.835
x4  = 25.436
x6  = 15.720
x8  = 12.258
x10 = 10.259
x12 =  8.823
x14 =  7.720
x16 =  6.748
x18 =  6.098
x20 =  5.596
x22 =  5.674
x24 =  5.354
x26 =  4.917
x28 =  4.599
x30 =  4.143
x32 =  3.846
x34 =  3.690
x36 =  3.554
x38 =  3.300
x40 =  3.227

x_task_lista = [x1,x2,x4,x6,x8,x10,x12,x14,x16,x18,x20,x22,x24,x26,x28,x30,x32,x34,x36,x38,x40]
ganho_tasks = [x1 / elem for elem in x_task_lista]

x2  = 49.379
x10 = 37.580
x20 = 34.734
x30 = 35.463
# x38 = 33.338
x40 = 33.324

x_thread_lista = [x1,x2,x10,x20,x30,x40]
ganho_thread = [x1 / elem for elem in x_thread_lista]

ganho_esperado = [1] + [value for value in range(2,42,2)]

x = [1] + [i for i in range(2,42,2)]
x2 = [1,2] + [i for i in range(10,50,10)]
points_ganho_tasks = list(zip(x,ganho_tasks))
points_ganho_thread = list(zip(x2,ganho_thread))
points_esperado = list(zip(x,ganho_esperado))
plot_points(points_ganho_tasks,points_ganho_thread,points_esperado)
