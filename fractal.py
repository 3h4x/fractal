#!/usr/bin/env python3
from __future__ import division, print_function
import numpy as np
import pylab, collections, functools
from itertools import takewhile
import time

Point = collections.namedtuple("Point", ["x", "y"])


def pair_reader(dtype):
    return lambda data: Point(*map(dtype, data.lower().split("x")))


DEFAULT_SIZE = "512x512"
DEFAULT_DEPTH = "256"
DEFAULT_ZOOM = "1"
DEFAULT_CENTER = "0x0"

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
CENTER_X = SCREEN_WIDTH / 2
CENTER_Y = SCREEN_HEIGHT / 2

max_iteration = 50
scale = 3.0 / (SCREEN_HEIGHT * 500.0)

def generate_fractal(
    model,
    c=None,
    size=pair_reader(int)(DEFAULT_SIZE),
    depth=int(DEFAULT_DEPTH),
    zoom=float(DEFAULT_ZOOM),
    center=pair_reader(float)(DEFAULT_CENTER),
):
    """
    2D Numpy Array with the fractal value for each pixel coordinate.
    """

    start = time.time()

    rows = [
        generate_row(model, c, size, depth, zoom, center, row) for row in range(size[1])
    ]

    # Generates the intensities for each pixel
    img = pylab.array(rows)

    print(f"Render zoom {zoom} in {time.time() - start:.2f}s")
    return img


def generate_row(model, c, size, depth, zoom, center, row):
    """
    Generate a single row of fractal values, enabling shared workload.
    """
    func = get_model(model, depth, c)
    width, height = size
    cx, cy = center
    side = max(width, height)
    sidem1 = side - 1
    deltax = (side - width) / 2  # Centralize
    deltay = (side - height) / 2
    y = (2 * (height - row + deltay) / sidem1 - 1) / zoom + cy
    return [
        func((2 * (col + deltax) / sidem1 - 1) / zoom + cx, y) for col in range(width)
    ]


def get_model(model, depth, c):
    """
    Returns the fractal model function for a single pixel.
    """
    if model == "julia":
        func = cqp(c)
        return lambda x, y: fractal_eta(x + y * 1j, func, depth)
    if model == "mandelbrot":
        return lambda x, y: fractal_eta(0, cqp(x + y * 1j), depth)
    raise ValueError("Fractal not found")


def repeater(f):
    """
    Returns a generator function that returns a repeated function composition
    iterator (generator) for the function given, i.e., for a function input
    ``f`` with one parameter ``n``, calling ``repeater(f)(n)`` yields the
    values (one at a time)::

       n, f(n), f(f(n)), f(f(f(n))), ...

    Examples
    --------

    >>> func = repeater(lambda x: x ** 2 - 1)
    >>> func
    <function ...>
    >>> gen = func(3)
    >>> gen
    <generator object ...>
    >>> next(gen)
    3
    >>> next(gen) # 3 ** 2 - 1
    8
    >>> next(gen) # 8 ** 2 - 1
    63
    >>> next(gen) # 63 ** 2 - 1
    3968

    """

    @functools.wraps(f)
    def wrapper(n):
        val = n
        while True:
            yield val
            val = f(val)

    return wrapper


def in_circle(radius):
    """Returns ``abs(z) < radius`` boolean value function for a given ``z``"""
    return lambda z: z.real**2 + z.imag**2 < radius**2


def fractal_eta(z, func, limit, radius=2):
    """
    Fractal Escape Time Algorithm for pixel (x, y) at z = ``x + y * 1j``.
    Returns the fractal value up to a ``limit`` iteration depth.
    """
    return amount(takewhile(in_circle(radius), repeater(func)(z)), limit)


def amount(gen, limit=float("inf")):
    """
    Iterates through ``gen`` returning the amount of elements in it. The
    iteration stops after at least ``limit`` elements had been iterated.

    Examples
    --------

    >>> amount(x for x in "abc")
    3
    >>> amount((x for x in "abc"), 2)
    2
    >>> from itertools import count
    >>> amount(count(), 5) # Endless, always return ceil(limit)
    5
    >>> amount(count(start=3, step=19), 18.2)
    19
    """
    size = 0
    for unused in gen:
        size += 1
        if size >= limit:
            break
    return size


def cqp(c):
    """Complex quadratic polynomial, function used for Mandelbrot fractal"""
    return lambda z: z**2 + c
