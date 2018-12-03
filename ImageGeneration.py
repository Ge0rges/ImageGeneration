from PIL import Image
from collections import defaultdict, Counter
import numpy as np
import pyprind
import random
import os
import pygame
import sys


class MarkovChain(object):
    def __init__(self, direction=True):
        self.weights = defaultdict(Counter)
        self.directional = direction
        self.normalize_alpha = (1, 1, 1)

    def set_normalize_alpha(self, pixels):
        """Calculates the normalization constant alpha for each RGB component over this image.
        Sets it in self.normalize_alpha.
        :param pixels A list of pixels (r, g, b)
        """

        sum_r = 0
        sum_g = 0
        sum_b = 0

        for r, g, b in pixels:
            sum_r += r
            sum_g += g
            sum_b += b

        self.normalize_alpha = (sum_r, sum_g, sum_b)

    def normalize(self, pixel):
        """ Divides each pixel's RGB values by the scalar to normalize the distribution.
        :param pixel A pixel (r, g, b)
        :return A normalized pixel (r, g, b)
        """

        return (pixel[0] / self.normalize_alpha[0],
                pixel[1] / self.normalize_alpha[1],
                pixel[2] / self.normalize_alpha[2])

    def denormalize(self, pixel):
        """ Multiplies each pixel's RGB values by the scalar to denormalize the distribution.
        :param pixel A pixel (r, g, b)
        :return A denormalized pixel (r, g, b)
        """
        return (pixel[0] * self.normalize_alpha[0],
                pixel[1] * self.normalize_alpha[1],
                pixel[2] * self.normalize_alpha[2])

    def get_neighbours(self, x, y):
        """Returns the 8 coordinates for each neighbour of the pixel at (x, y).
        :param x X coordinate of the pixel
        :param y Y coordinate of the pixel
        :return The coordinates of (x, y)'s neighbour in a list"""
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                (x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1)]

    def get_neighbours_direction(self, x, y):
        """Returns the 8 coordinates for each neighbour of the pixel at (x, y) in a dictionary matching each coordinate
        to the direction relative to the pixel.
        :param x X coordinate of the pixel
        :param y Y coordinate of the pixel
        :return The coordinates of (x, y)'s neighbour as a mapped dictionary"""
        return {'r': (x + 1, y),
                'l': (x - 1, y),
                'b': (x, y + 1),
                't': (x, y - 1),
                'br': (x + 1, y + 1),
                'tl': (x - 1, y - 1),
                'tr': (x - 1, y + 1),
                'bl': (x + 1, y - 1)}

    def train(self, image):
        """
        Train on the input PIL image
        :param image: The input image
        """
        width, height = image.size
        image = np.array(image)[:, :, :3]
        prog = pyprind.ProgBar(width * height * 2, title="Training", width=64, stream=1)

        pixels = []
        for x in range(height):
            for y in range(width):
                pixels.append(image[x, y])
                prog.update()

        self.set_normalize_alpha(pixels)

        for x in range(height):
            for y in range(width):
                # get the left, right, top, bottom neighbour pixels
                pix = tuple(self.normalize(image[x, y]))
                prog.update()
                for neighbour in self.get_neighbours(x, y):
                    try:
                        self.weights[pix][tuple(self.normalize(image[neighbour]))] += 1
                    except IndexError:
                        continue

        self.directional = False

    def train_direction(self, image):
        """
        Train on the input PIL image with direction.
        :param image: The input image
        """

        self.weights = defaultdict(lambda: defaultdict(Counter))
        width, height = image.size
        image = np.array(image)[:, :, :3]
        prog = pyprind.ProgBar(width * height * 2, title="Training", width=64, stream=1)

        pixels = []
        for x in range(height):
            for y in range(width):
                pixels.append(image[x, y])
                prog.update()

        self.set_normalize_alpha(pixels)

        for x in range(height):
            for y in range(width):
                pix = tuple(self.normalize(image[x, y]))
                prog.update()
                for direction, neighbour in self.get_neighbours_direction(x, y).items():
                    try:
                        self.weights[pix][dir][tuple(self.normalize(image[neighbour]))] += 1

                    except IndexError:
                        continue

        self.directional = True

    def generate(self, initial_state=None, width=512, height=512):
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Markov Image')
        screen.fill((0, 0, 0))

        if initial_state is None:
            initial_state = random.choice(list(self.weights.keys()))

        if type(initial_state) is not tuple and len(initial_state) != 3:
            raise ValueError("Initial State must be a 3-tuple")

        image = Image.new('RGB', (width, height), 'white')
        image = np.array(image, dtype=np.double)
        image_out = np.array(image.copy(), dtype=np.uint8)

        # start filling out the image
        # start at a random point on the image, set the neighbours and then move into a random, unchecked neighbour,
        # only filling in unmarked pixels
        initial_position = (np.random.randint(0, width), np.random.randint(0, height))
        image[initial_position] = initial_state
        stack = [initial_position]
        coloured = set()
        prog = pyprind.ProgBar(width * height, title="Generating", width=64, stream=1)

        i = 0
        while stack:
            x, y = stack.pop()
            if (x, y) in coloured:
                continue
            else:
                coloured.add((x, y))

            try:
                cpixel = image[x, y]
                node = self.weights[tuple(cpixel)]  # a counter of neighbours

                cpixel = self.denormalize(cpixel)
                image_out[x, y] = (round(cpixel[0]).astype(np.uint8),
                                   round(cpixel[1]).astype(np.uint8), round(cpixel[2]).astype(np.uint8))

                prog.update()

                # Update the display live every 128 iterations
                i += 1
                screen.set_at((x, y), image_out[x, y])
                if i % 128 == 0:
                    pygame.display.flip()
                    pass

            except IndexError:
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            if self.directional:
                keys = {direction: list(node[direction].keys()) for direction in node}
                neighbours = self.get_neighbours_direction(x, y).items()
                counts = {direction: np.array(list(node[direction].values()), dtype=np.float32) for direction in keys}
                key_idxs = {direction: np.arange(len(node[direction])) for direction in keys}
                ps = {direction: counts[direction] / counts[direction].sum() for direction in keys}

            else:
                keys = list(node.keys())
                neighbours = self.get_neighbours(x, y)
                counts = np.array(list(node.values()), dtype=np.float32)
                key_idxs = np.arange(len(keys))
                ps = counts / counts.sum()
                np.random.shuffle(neighbours)

            for neighbour in neighbours:
                try:
                    if self.directional:
                        direction = neighbour[0]
                        neighbour = neighbour[1]
                        if neighbour not in coloured:
                            col_idx = np.random.choice(key_idxs[direction], p=ps[direction])
                            image[neighbour] = keys[direction][col_idx]
                    else:
                        col_idx = np.random.choice(key_idxs, p=ps)
                        if neighbour not in coloured:
                            image[neighbour] = keys[col_idx]

                except IndexError:
                    pass

                except ValueError:
                    continue

                if 0 <= neighbour[0] < width and 0 <= neighbour[1] < height:
                    stack.append(neighbour)

        return Image.fromarray(image_out)


if __name__ == "__main__":
    chain = MarkovChain(direction=False)

    file_names = ['test1.png']
    if len(sys.argv) > 1:
        fnames = sys.argv[1:]

    for file_name in file_names:
        image = Image.open(file_name)

        if chain.directional:
            chain.train_direction(image)
        else:
            chain.train(image)

    chain.generate().show()
