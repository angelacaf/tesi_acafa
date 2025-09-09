import matplotlib.pyplot as plt
import numpy as np
import os

MODE='SAVE'
COLOR_SPACE = {'trajectory': 'b', 'outter_bound': 'g', 'inner_bound': 'r', 'projection_line': 'orange', 'projection_line':"#7E2F8E", 'spline_ctrl': 'magenta'
}


# The image array saves the pixel in coordiate of [y, x]
def show_image(image_array, title='image'):
    shape = np.shape(image_array)
    plt.figure()
    plt.imshow(np.transpose(image_array), cmap='gray')
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])

    if title:
        plt.title(title)
    if MODE == 'SAVE':
        plt.savefig(f'./figures/{title}.png')
    else:
        plt.show()


# The image array saves the pixel in coordiate of [y, x]
def save_debug_image(image_array, save_path):
    shape = np.shape(image_array)
    plt.figure()
    plt.imshow(np.transpose(image_array), cmap='gray')
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])
    plt.savefig(save_path)
    plt.close()


def show_curves_on_image(curve_info_list, image_array, save_path, point_size = 1):
    shape = np.shape(image_array)
    plt.figure()
    plt.imshow(np.transpose(image_array), cmap='gray')
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])

    for curve_info in curve_info_list:
        xs = curve_info['xs']
        ys = curve_info['ys']
        color = COLOR_SPACE.get(curve_info['type'], 'yellow')
        plt.plot(xs, ys, color, linewidth=point_size)
    plt.savefig(save_path)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

def save_trajectory_on_image(
    trajectory,
    image_array,
    save_path_base='spline',
    resolution=0.1,
    ctrl_points=None,
    dpi=300
):
    xs, ys = trajectory.get_points_array_on_trajectory(resolution)

    # --- 1. Con immagine MIP sotto ---
    plt.figure()
    plt.imshow(np.transpose(image_array), cmap='gray')
    plt.xlim(0, image_array.shape[0])
    plt.ylim(0, image_array.shape[1])
    plt.plot(xs, ys, color='blue', linewidth=1.5, label='Spline')

    if ctrl_points is not None:
        ctrl_points = np.array(ctrl_points)
        plt.scatter(ctrl_points[:, 0], ctrl_points[:, 1], color='red', s=10, label='Control Points')

    plt.legend()
    plt.axis('equal')
    plt.savefig(f'{save_path_base}_with_image.png', dpi=dpi)
    plt.savefig(f'{save_path_base}_with_image.svg', dpi=dpi)
    plt.close()

    # --- 2. Solo spline + punti di controllo, sfondo trasparente ---
    plt.figure()
    plt.xlim(0, np.max(xs) + 10)
    plt.ylim(0, np.max(ys) + 10)
    plt.gca().invert_yaxis()  # Per orientamento medico
    plt.plot(xs, ys, color='blue', linewidth=1.5, label='Spline')

    if ctrl_points is not None:
        ctrl_points = np.array(ctrl_points)
        plt.scatter(ctrl_points[:, 0], ctrl_points[:, 1], color='red', s=10, label='Control Points')

    plt.legend()
    plt.axis('equal')
    plt.axis('off')  # Niente assi o ticks
    plt.savefig(f'{save_path_base}_only_curve.png', dpi=dpi, transparent=True)
    plt.savefig(f'{save_path_base}_only_curve.svg', dpi=dpi, transparent=True)
    plt.close()

