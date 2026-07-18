import os
from PIL import Image
import matplotlib.pyplot as plt


def display_images(image_paths, labels=None, figsize=(12, 3), num_images=4):
    """Load 4 images and display them side-by-side with overhead labels."""
    if len(image_paths) != num_images:
        raise ValueError(f"image_paths must contain exactly {num_images} image files.")

    if labels is None:
        labels = [os.path.basename(path) for path in image_paths]

    if len(labels) != num_images:
        raise ValueError("labels must contain exactly 4 items.")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    target_size = images[0].size
    images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]

    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    paths = [
        './sample_explanations/cifar10_test_image.png',
        './sample_explanations/gradcam_example.jpg',
        './sample_explanations/lime_example.jpg',
        './sample_explanations/ig_example.jpg'
    ]

    labels = ['Original', 'GradCAM', 'LIME', 'Integrated Gradients']

    display_images(image_paths=paths, labels=labels, figsize=(8, 3), num_images=len(paths))

