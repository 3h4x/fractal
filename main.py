#!/usr/bin/env python3
from __future__ import division, print_function
import pygame
import numpy as np

from fractal import DEFAULT_SIZE, generate_fractal, pair_reader

pygame.init()
# screen = pygame.display.set_mode(pair_reader(int)(DEFAULT_SIZE), pygame.FULLSCREEN)
screen = pygame.display.set_mode(pair_reader(int)(DEFAULT_SIZE))


def main():
    done = False
    clock = pygame.time.Clock()
    zoom = 90
    render = True

    while not done:
        # This limits the while loop to a max of 60 times per second.
        # Leave this out and we will use all CPU we can.
        clock.tick(20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: # If user clicked close
                done = True # Flag that we are done so we exit this loop
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    zoom -= 10
                    render = True
                if event.key == pygame.K_DOWN:
                    zoom += 10
                    render = True

        if render:
          draw_fractal(screen, zoom)
          render = False
        pygame.display.flip()

    pygame.quit()


def draw_fractal(screen, zoom=90):
    depth = 256
    center = "-1.255x0.38"
    img = generate_fractal(
        "mandelbrot",
        c=1,
        size=pair_reader(int)(DEFAULT_SIZE),
        depth=depth,
        zoom=zoom,
        center=pair_reader(float)(center),
    )

    iterator = np.nditer(img, flags=["multi_index"])
    for value in iterator:
        pygame.draw.line(
            screen,
            min(255, value.item())*np.array([1,1,1]),  # type: ignore
            (iterator.multi_index[0], iterator.multi_index[1]),
            (iterator.multi_index[0], iterator.multi_index[1]),
        )

main()
