"""
Python script to create an animation from a series of images stored alphabetically in a directory.

The structure of image directory should be like
```
image_dir
├── image_grid_8x8_level_000_step_000.pth
├── image_grid_8x8_level_000_step_001.pth
├── ...
├── image_grid_8x8_level_095_step_000.pth
├── image_grid_8x8_level_095_step_000.pth
└── image_grid_final_denoised.pth
```
The used `.pth` files should have their name start with `image_grid` and end with `.pth`.
If the name does not follow this format, they won't be included in the animation.
Each used `.pth` file should contain an image grid of shape (H, W, C) stored as a PyTorch tensor.
Note that the files will be sorted in alphabetical order based on their filenames,
so make sure the filenames reflect the desired order of frames in the animation.

After running the script, an animation gif will be saved to the specified path.

To create an animation, run the following command in the terminal:
```bash
python make_animation.py -I [path/to/image_dir] -O [path/to/save/animation.gif]
```
For example:
```bash
python make_animation.py -I exp/experiment_1760865815_2.0_0.5_4.5_k_p=2.0_k_i=0.5_k_d=4.5_100x1_steps/image_samples -O animation.gif
```
"""
import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_image_animation(fig, ax, images, frame_text_func=lambda frame: f"Frame {frame}"):
    """
    Make an animation of a set of images over time.

    Args:
        fig (matplotlib.figure.Figure): The figure object to plot the animation on.
        ax (matplotlib.axes.Axes): The axes object to plot the animation on.
        images (np.ndarray): An array of shape (T, height, width, channels) containing the images in each frame.
        frame_text_func (function): A function that takes a frame number (int) and returns the text to display for that frame.
    
    Returns:
        ani (matplotlib.animation.FuncAnimation): The animation object.

    Examples:
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        images = np.random.randn(200, 64, 64, 3) # Generate 200 frames of 64x64 RGB images
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        frame_text_func = lambda frame: f"Frame {frame}/{len(images)}"
        ani = make_image_animation(fig, ax, images, frame_text_func=frame_text_func)
        save_path = 'animation222.gif'
        ani.save(save_path, writer='pillow', fps=30) # Save animation as gif
        plt.show()
        ```
    """
    im = ax.imshow(images[0])

    frame_text = ax.text(0.5, 0.95, "",
                        transform=ax.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')

    def init():
        im.set_data(images[0])
        return im,

    def update(frame):
        im.set_data(images[frame]) # Update image object with current frame data
        frame_text.set_text(frame_text_func(frame))
        
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(images), init_func=init, blit=True)

    return ani


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-I', '--image_dir', type=str, required=True, help="Path to the image directory")
    parser.add_argument('-O', '--save_path', type=str, default=None, help="Path to save the animation gif")
    parser.add_argument('-F', '--fps', type=int, default=4, help="Frames per second for the animation gif")
    
    args = parser.parse_args()

    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = 'animation.gif'
    if os.path.exists(save_path):
        print(f"Warning: {save_path} already exists. Do you want to overwrite? [ Y / N ]")
        choice = input()
        if choice != 'Y':
            print("Program exitted.")
            return

    image_dir = args.image_dir
    image_paths = [p for p in os.listdir(image_dir) if p.endswith('pth') and p.startswith('image_grid')]

    images=np.array([torch.load(os.path.join(image_dir, image_path)) for image_path in image_paths])
    print("Images shape: {}.".format(images.shape)) # (T, H, W, C)

    fig, ax = plt.subplots(figsize=(6,6))
    frame_text_func = lambda frame: f""
    ani = make_image_animation(fig, ax, images, frame_text_func=frame_text_func)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    ani.save(save_path, writer='pillow', fps=args.fps) # Save animation as gif
    plt.close()

    print(f"Done. Animation saved to {save_path}.")


if __name__ == '__main__':
    main()
