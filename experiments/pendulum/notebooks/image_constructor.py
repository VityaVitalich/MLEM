import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import asyncio
import pickle
from multiprocessing import Pool
import time
from tqdm.contrib.concurrent import process_map

np.random.seed(42)


def hawkes_intensity(mu, alpha, points, t):
    """Find the hawkes intensity:
    mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )
    """
    p = np.array(points)
    p = p[p <= t]
    p = np.exp(p - t) * alpha
    return mu + np.sum(p)
    # return mu + alpha * sum( np.exp(s - t) for s in points if s <= t )


assert np.isclose(1, hawkes_intensity(1, 2, [], 5))
assert np.isclose(2, hawkes_intensity(0, 2, [5], 5))
assert np.isclose(4, hawkes_intensity(2, 2, [4.9999999], 5))
assert np.isclose(2, hawkes_intensity(0, 2, [4.9999999, 8], 5))
assert np.isclose(1, hawkes_intensity(1, 2, [5, 8], 4))


def simulate_hawkes(mu, alpha, st, et):
    t = st
    points = []
    all_samples = []
    while t < et:
        m = hawkes_intensity(mu, alpha, points, t)
        s = np.random.exponential(scale=1 / m)
        ratio = hawkes_intensity(mu, alpha, points, t + s) / m
        if t + s >= et:
            break
        if ratio >= np.random.uniform():
            points.append(t + s)
        all_samples.append(t + s)
        t = t + s
    return points, all_samples


def create_sample(L):
    # https://skill-lync.com/student-projects/Simulation-of-a-Simple-Pendulum-on-Python-95518

    def sim_pen_eq(t, theta):
        dtheta2_dt = (-b / m) * theta[1] + (-g / L) * np.sin(theta[0])
        dtheta1_dt = theta[1]
        return [dtheta1_dt, dtheta2_dt]

    # main

    theta1_ini = np.random.uniform(1, 9)  # Initial angular displacement (rad)
    theta2_ini = np.random.uniform(1, 9)  # Initial angular velocity (rad/s)
    theta_ini = [theta1_ini, theta2_ini]
    t_span = [st, et + ts]

    points, all_samples = simulate_hawkes(mu, alpha, st, et)
    t_ir = points
    sim_points = len(points)

    theta12 = solve_ivp(sim_pen_eq, t_span, theta_ini, t_eval=t_ir)
    theta1 = theta12.y[0, :]
    theta2 = theta12.y[1, :]

    # return x, y
    # or we could return angles ...
    x = L * np.sin(theta1)
    y = -L * np.cos(theta1)

    return x, y, t_ir


# Initial and end values
st = 0  # Start time (s)
et = 5  # End time (s)
ts = 0.1  # Time step (s)
g = 9.81  # Acceleration due to gravity (m/s^2)
b = 0.5  # Damping factor (kg/s)
m = 1  # Mass of bob (kg)
mu = 10
alpha = 0.2


def create_dataset(size):
    output = []

    for i in tqdm(range(size)):
        L = np.random.uniform(0.5, 5)
        sample = create_sample(L)
        output.append((sample, L))

    return output


def point_to_image(x, y, L):
    arrs = []
    for point in range(len(x)):
        plt.figure(figsize=(1.6, 1.6))  # 16x16 pixels at 100 dpi
        plt.plot(x[point], y[point], "bo", markersize=8)
        # plt.plot([0, x[point]], [0, y[point]])
        plt.xlim(-L - 0.5, L + 0.5)
        plt.ylim(-L - 0.5, L + 0.5)
        plt.axis("off")  # Turn off axes
        # Save the plot as a PNG file with a specific dpi
        plt.savefig(f"imgs/plot_{point}_{L}.png", dpi=10)
        plt.close()
        # Load the saved image using PIL and convert it to black and white
        img = Image.open(f"imgs/plot_{point}_{L}.png").convert("1")
        img_array = np.array(img)
        arrs.append(img_array.flatten())
        os.remove(f"imgs/plot_{point}_{L}.png")
    return arrs


def get_arrs(obj):
    x, y, t = obj[0]
    l = obj[1]
    return point_to_image(x, y, l)


if __name__ == "__main__":
    ds = create_dataset(10000)
    start = time.time()
    # with Pool() as p:
    #     out = list(tqdm(p.imap(get_arrs, ds), total=50))
    out = []

    for i in range(1, 10):
        cur_min = (i - 1) * 1000
        cur_max = i * 1000
        out.extend(process_map(get_arrs, ds[cur_min:cur_max], chunksize=1))

    # out = get_all_arrs(ds)
    end = time.time()
    print(end - start)
    name = "../data/hawkes_16.pickle"

    with open(name, "wb") as f:
        pickle.dump(out, f)
