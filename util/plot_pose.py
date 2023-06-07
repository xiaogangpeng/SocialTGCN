
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import imageio
from PIL import Image


def plot_pose(pose, cur_frame, prefix, scene):
    body_edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
         [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [11, 15], [15, 16], [16, 17]]
    )
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111, projection='3d')
    
    if scene =='Complex_Crowd':
        scale = 14
    elif scene =='Special_Locations':
        scale = 7
    else:
        scale = 5

    p_x = np.linspace(-scale, scale, scale*2)
    p_y = np.linspace(-scale, scale, scale*2)
   
    X, Y = np.meshgrid(p_x, p_y)

    for x_i in range(p_x.shape[0]):
        temp_x = [p_x[x_i], p_x[x_i]]
        temp_y = [p_y[0], p_y[-1]]
        z = [0, 0]
        ax.plot(temp_x, temp_y, z, color='black', alpha=0.1)

    for y_i in range(p_x.shape[0]):
        temp_x = [p_x[0], p_x[-1]]
        temp_y = [p_y[y_i], p_y[y_i]]
        z = [0, 0]
        ax.plot(temp_x, temp_y, z, color='black', alpha=0.1)
    # print(f"Test: {pose.shape} ")
    for j in range(pose.shape[0]):
        xs = pose[j, cur_frame, :, 0]
        ys = pose[j, cur_frame, :, 1]
        zs = pose[j, cur_frame, :, 2]

        alpha = 0.6
        ax.scatter(xs, ys, zs, c='#000', marker="o", s=1.0, alpha=alpha)
        plot_edge = True
        if plot_edge:
            for edge in body_edges:
                alpha2 = 1
                x = [pose[j, cur_frame, edge[0], 0], pose[j, cur_frame, edge[1], 0]]
                y = [pose[j, cur_frame, edge[0], 1], pose[j, cur_frame, edge[1], 1]]
                z = [pose[j, cur_frame, edge[0], 2], pose[j, cur_frame, edge[1], 2]]
                if cur_frame < 25:
                    ax.plot(x, y, z,  '#eeeeee',  linewidth='2', alpha = 0.8)
                    ax.plot(x, y, z, '#000000', linewidth='.5', alpha =0.8)
                else:                
                    if j == 0:
                        ax.plot(x, y, z, '#4F9DA6', linewidth='2', alpha = 0.8)
                        ax.plot(x, y, z, '#000000', linewidth='.5', alpha =0.8)
                    elif j == 1:
                        ax.plot(x, y, z, '#FFAD5A', linewidth='2', alpha = 0.8)
                        ax.plot(x, y, z, '#000', linewidth='0.5', alpha =0.8)
                    elif j == 2:
                        ax.plot(x, y, z, '#FF5959', linewidth='2', alpha = 0.8)
                        ax.plot(x, y, z, '#000', linewidth='.5', alpha =0.8)
                    elif j == 3:
                        ax.plot(x, y, z, '#00716F', linewidth='2', alpha=0.8)
                        ax.plot(x, y, z, '#000', linewidth='.5', alpha=0.8)
                    elif j == 4:
                        ax.plot(x, y, z, '#2D76A8', linewidth='2', alpha=0.8)
                        ax.plot(x, y, z, '#000', linewidth='.5', alpha=0.8)
                    elif j == 5:
                        ax.plot(x, y, z, '#FF6C5B', linewidth='2', alpha=0.8)
                        ax.plot(x, y, z, '#000', linewidth='.5', alpha=0.8)
        plt.title('Frame: '+str(cur_frame), y=-0.1)            
        ax.set_xlim3d([-5, 5])
        ax.set_ylim3d([-5, 5])
        ax.set_zlim3d([0, 5])
        ax.elev = 35
        ax.azim = 15
        ax.set_axis_off()

    plt.draw()
    plt.savefig(prefix + '_'+ str(cur_frame) + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_imgs(batch_i, method, scene, pred_poses, gt_poses, input_poses, opt):
    gif_dir = 'baselines/'+str(method)+'/outputs/test_gif/' + str(scene)
    img_dir = 'baselines/'+str(method)+'/outputs/test_img/' + str(scene)  
    pred_seq = pred_poses.squeeze()
    gt_seq = gt_poses.squeeze()
    input_seq = input_poses.squeeze()
    pred = torch.cat([input_seq, pred_seq], dim=1).detach().cpu().numpy()
    gt = torch.cat([input_seq, gt_seq], dim=1).detach().cpu().numpy()


    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    pred_img_list = []
    gt_img_list = []
    for t in range(pred.shape[1]):
        plot_pose(pred, t, img_dir + '/pred_batch_' + str(batch_i), scene)
        plot_pose(gt, t, img_dir + '/gt_batch_' + str(batch_i), scene)
        pred_img = Image.open(img_dir + '/pred_batch_' + str(batch_i) + '_' + str(t) + '.png', 'r')
        gt_img = Image.open(img_dir + '/gt_batch_' + str(batch_i) + '_' + str(t) + '.png', 'r')
        pred_img_list.append(pred_img)
        gt_img_list.append(gt_img)
    imageio.mimsave((gif_dir + '/pred_img_%03d.gif' % batch_i), pred_img_list, duration=0.12)
    imageio.mimsave((gif_dir + '/gt_img_%03d.gif' % batch_i), gt_img_list, duration=0.12)