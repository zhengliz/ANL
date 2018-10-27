import cv2, os, shutil
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from maze import *


c_orientations = ['East', 'North', 'West', 'South']
c_colors = ['Greys', 'Greens', 'Blues', 'Purples']
c_angles = [-90, 0, 90, 180]
c_actions = ['left', 'right', 'forward']


def clear_folder(fpath):
    shutil.rmtree(fpath)
    os.makedirs(fpath)


def visualize(map_design, belief, position=None, orientation=None, fn=None, idx=None):
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=[8, 2])
    for a in ax:
        # a.axis('off')
        a.set_xticks([])
        a.set_yticks([])
    if map_design.ndim == 2:
        for i in range(4):
            ax[i].imshow(belief[i], cmap=c_colors[i], vmin=0, vmax=1)
            ax[i].set_title(c_orientations[i])

        ax[4].imshow(map_design, cmap='binary', vmin=0, vmax=1)
        ax[4].set_title('map')
        if position:
            ax[4].plot(position[1], position[0], marker=(3, 0, c_angles[orientation]), markersize=8, color='r')

    elif map_design.ndim == 3:
        pass

    if idx is not None:
        fig.suptitle('FRAME {}'.format(idx))
    if fn:
        fig.savefig(fn)
    # plt.show()
    # return fig
    plt.close()


def visualize_comparison(map_design, belief_pair, position_pair, orientation_pair, fn=None, idx=None):
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=[8, 6])
    for x in ax:
        for a in x:
            a.set_xticks([])
            a.set_yticks([])
    ax[1, 4].axis('off')

    diffs = belief_pair[1] - belief_pair[0]

    if map_design.ndim == 2:
        limit = max(abs(diffs).max(), 0.01)
        for i in range(4):
            ax[0, i].imshow(belief_pair[0][i], cmap=c_colors[i], vmin=0, vmax=1)
            ax[0, i].set_title(c_orientations[i])

            ax[1, i].imshow(diffs[i], cmap='RdBu', vmin=-limit, vmax=limit)

            ax[2, i].imshow(belief_pair[1][i], cmap=c_colors[i], vmin=0, vmax=1)

        ax[0, 4].imshow(map_design, cmap='binary', vmin=0, vmax=1)
        ax[0, 4].set_title('map')
        ax[2, 4].imshow(map_design, cmap='binary', vmin=0, vmax=1)

        ax[0, 0].set_ylabel('origin')
        ax[1, 0].set_ylabel('diff')
        ax[2, 0].set_ylabel('perturb')

        if position_pair:
            ax[0, 4].plot(position_pair[0][1], position_pair[0][0],
                          marker=(3, 0, c_angles[orientation_pair[0]]),
                          markersize=8, color='r')
            ax[2, 4].plot(position_pair[1][1], position_pair[1][0],
                          marker=(3, 0, c_angles[orientation_pair[1]]),
                          markersize=8, color='r')

    elif map_design.ndim == 3:
        pass

    if idx is not None:
        fig.suptitle('FRAME {}'.format(idx))
    if fn:
        fig.savefig(fn)
    # plt.show()
    # return fig
    plt.close()


def images2video(fin, fout):
    images = os.listdir(fin)
    images.sort()
    # Determine the width and height from the first image
    image_path = os.path.join(fin, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(fout, fourcc, 1.0, (width, height))

    for image in images:

        image_path = os.path.join(fin, image)
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()
    print("The output video is {}".format(fout))


if __name__ == '__main__':
    map_design = generate_map(7)
    position = (1, 4)
    orientation = 3

    belief = np.random.randn(4, 7, 7)
    belief /= belief.sum()

    visualize(map_design, belief, position, orientation, './test.png')
