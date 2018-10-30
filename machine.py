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


def softmax_convert_into_pi_from_theta(theta):
    '''ソフトマックス関数で割合を計算する'''

    beta = 1.0
    [m, n] = theta.shape  # thetaの行列サイズを取得
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta)  # thetaをexp(theta)へと変換

    for i in range(0, m):
        # pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
        # simpleに割合の計算の場合

        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        # softmaxで計算の場合

    pi = np.nan_to_num(pi)  # nanを0に変換

    return pi


def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    # pi[s,:]の確率に従って、directionが選択される
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s - 3  # 上に移動するときは状態の数字が3小さくなる
    elif next_direction == "right":
        action = 1
        s_next = s + 1  # 右に移動するときは状態の数字が1大きくなる
    elif next_direction == "down":
        action = 2
        s_next = s + 3  # 下に移動するときは状態の数字が3大きくなる
    elif next_direction == "left":
        action = 3
        s_next = s - 1  # 左に移動するときは状態の数字が1小さくなる

    return [action, s_next]


def goal_maze_ret_s_a(pi):
    s = 0  # スタート地点
    s_a_history = [[0, np.nan]]  # エージェントの移動を記録するリスト

    while (1):  # ゴールするまでループ
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        # 現在の状態（つまり一番最後なのでindex=-1）の行動を代入

        s_a_history.append([next_s, np.nan])
        # 次の状態を代入。行動はまだ分からないのでnanにしておく

        if next_s == 8:  # ゴール地点なら終了
            break
        else:
            s = next_s

    return s_a_history


def update_theta(theta, pi, s_a_history):
    eta = 0.1  # 学習率
    T = len(s_a_history) - 1  # ゴールまでの総ステップ数

    [m, n] = theta.shape  # thetaの行列サイズを取得
    delta_theta = theta.copy()  # Δthetaの元を作成、ポインタ参照なので、delta_theta = thetaはダメ

    # delta_thetaを要素ごとに求めます
    for i in range(0, m):
        for j in range(0, n):
            if not (np.isnan(theta[i, j])):  # thetaがnanでない場合

                SA_i = [SA for SA in s_a_history if SA[0] == i]
                # 履歴から状態iのものを取り出すリスト内包表記です

                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                # 状態iで行動jをしたものを取り出す

                N_i = len(SA_i)  # 状態iで行動した総回数
                N_ij = len(SA_ij)  # 状態iで行動jをとった回数

                # 初版では符号の正負に間違いがありました（修正日：180703）
                # delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T
                delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T

    new_theta = theta + eta * delta_theta

    return new_theta


def solve_labyrinth(theta_0):
    stop_epsilon = 10 ** -4  # 10^-4よりも方策に変化が少なくなったら学習終了とする

    theta = theta_0
    pi = softmax_convert_into_pi_from_theta(theta_0)

    is_continue = True
    count = 1
    while is_continue:  # is_continueがFalseになるまで繰り返す
        s_a_history = goal_maze_ret_s_a(pi)  # 方策πで迷路内を探索した履歴を求める
        new_theta = update_theta(theta, pi, s_a_history)  # パラメータΘを更新
        new_pi = softmax_convert_into_pi_from_theta(new_theta)  # 方策πの更新
        if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
            is_continue = False
        else:
            theta = new_theta
            pi = new_pi
    return s_a_history


def make_animation(fig, line, s_a_history):
    def init():
        # 背景画像の初期化
        line.set_data([], [])
        return (line,)

    def animate(i):
        # フレームごとの描画内容
        state = s_a_history[i][0]  # 現在の場所を描く
        x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
        y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
        line.set_data(x, y)
        return (line,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(
        s_a_history), interval=200, repeat=False)
    anim.save('output.gif', writer='imagemagick')

"""実行部"""
[line, theta_0, fig] = first_describe()
s_a_history = solve_labyrinth(theta_0)
make_animation(fig, line, s_a_history)
