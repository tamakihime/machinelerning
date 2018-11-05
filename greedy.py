# 使用するパッケージの宣言
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def first_describe():
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # 赤い壁を描く
    plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    # 状態を示す文字S0～S8を描く
    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    # 描画範囲の設定と目盛りを消す設定
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')

    # 現在地S0に緑丸を描画する
    line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
    theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                        [np.nan, 1, np.nan, 1],  # s1
                        [np.nan, np.nan, 1, 1],  # s2
                        [1, 1, 1, np.nan],  # s3
                        [np.nan, np.nan, 1, 1],  # s4
                        [1, np.nan, np.nan, np.nan],  # s5
                        [1, np.nan, np.nan, np.nan],  # s6
                        [1, 1, np.nan, np.nan],  # s7、※s8はゴールなので、方策はなし
                        ])
    return [line, theta_0, fig]


line, theta_0, fig = first_describe()


def get_Q(theta_0):
    [a, b] = theta_0.shape
    Q = np.random.rand(a, b) * theta_0
    return Q


def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi


pi_0 = simple_convert_into_pi_from_theta(theta_0)


def get_action(s, Q, epsion, pi_0):
    derection = ["up", "right", "down", "left"]

    if np.random.rand() < epsion:
        next_direction = np.random.choice(derection, p=pi_0[s, :])
    else:
        next_direction = derection[np.nanargmax(Q[s, :])]
    if next_direction == "up":
        action = 0
    if next_direction == "right":
        action = 1
    if next_direction == "down":
        action = 2
    if next_direction == "left":
        action = 3
    return action


def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a]
    if next_direction == "up":
        s_next = s - 3
    elif next_direction == "right":
        s_next = s + 1
    elif next_direction == "left":
        s_next = s + 3

    elif next_direction == "left":
        s_next = s = 1

    return s_next


def sarsa(s, a, r, s_next, a_next, Q, eta, gamma):
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])
        return Q


def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 0
    a = a_next = get_action(s,Q,epsilon,pi)
