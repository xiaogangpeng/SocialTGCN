import matplotlib.pyplot as plt
import numpy as np

input_path = 'MI-Motion/data_test_S0.npy'
data_list = []
data = np.load(input_path)
print(f"len :{data.shape}")

prefix = 20
data_list = data[prefix,:,:,:,:]



body_edges = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5],
    [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11],
    [11, 12], [12, 13], [13, 14], [11, 15], [15, 16], [16, 17]]
)



length_ = data_list.shape[1]

i = 0
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(111, projection='3d')

p_x = np.linspace(-10, 10, 10)
p_y = np.linspace(-10, 10, 10)
X, Y = np.meshgrid(p_x, p_y)
fig.tight_layout()


plt.ion()
#     temp_x = [p_x[x_i], p_x[x_i]]
#     temp_y = [p_y[0], p_y[-1]]
#     z = [0, 0]
#     ax.plot(temp_x, temp_y, z, color='black', alpha=0.1)
#
# for y_i in range(p_x.shape[0]):
#     temp_x = [p_x[0], p_x[-1]]
#     temp_y = [p_y[y_i], p_y[y_i]]
#     z = [0, 0]
#     ax.plot(temp_x, temp_y, z, color='black', alpha=0.1)

while i < length_:

    for j in range(data_list.shape[0]):
        xs = data_list[j, i, :, 0]
        ys = data_list[j, i, :, 1]
        zs = data_list[j, i, :, 2]
        # print(xs)
        alpha = 0.8
        ax.scatter(xs, ys, zs, c='#000', marker="o", s=4.0, alpha=alpha)
        plot_edge = True
        if plot_edge:
            for edge in body_edges:
                x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                if j == 0:
                    ax.plot(x, y, z, '#4F9DA6', linewidth='3.5', alpha = 0.8)
                    ax.plot(x, y, z, '#000000', linewidth='.8', alpha =0.8)
                elif j == 1:
                    ax.plot(x, y, z, '#FFAD5A', linewidth='3.5', alpha = 0.8)
                    ax.plot(x, y, z, '#000', linewidth='0.8', alpha =0.8)
                elif j == 2:
                    ax.plot(x, y, z, '#FF5959', linewidth='3.5', alpha = 0.8)
                    ax.plot(x, y, z, '#000', linewidth='.8', alpha =0.8)
        ax.set_xlim3d([-5, 5])
        ax.set_ylim3d([-5, 5])
        ax.set_zlim3d([0, 5])
        ax.elev = 30
        ax.azim = 15
        # ax.set_axis_off()
        # ax.set_xlim3d([-3, 3])
        # ax.set_ylim3d([-3, 3])
        # ax.set_zlim3d([0, 3])
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        plt.title(str(i), y=-0.01)
    plt.pause(0.0001)
    ax.cla()
    i += 1

plt.ioff()
plt.show()
plt.close()

# if j == 0:
#     ax.plot(x, y, z, '#6492BE', linewidth='5.0', alpha=0.4)
#     ax.plot(x, y, z, '#6492BE', linewidth='1.5')
# elif j == 1:
#     ax.plot(x, y, z, '#A8C77B', linewidth='5.0', alpha=0.4)
#     ax.plot(x, y, z, '#A8C77B', linewidth='1.5')
# elif j == 2:
#     ax.plot(x, y, z, '#E4A352', linewidth='5.0', alpha=0.4)
#     ax.plot(x, y, z, '#E4A352', linewidth='1.5')