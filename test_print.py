# import math
# # circumference_m = 0.229
# # radius_m = circumference_m / (2 * math.pi)
# # print(radius_m)

# # radius_m = 0.037
# # circumference_m = radius_m * 2 * math.pi
# # print(circumference_m)
# cl0 = 0.583  # 基本揚力係数
# cl1 = 2.333  # 揚力係数パラメータ1
# cl2 = 1.12  # 揚力係数パラメータ2

# rpm_to_rad_per_sec = math.pi / 30.0
# rad_per_sec_to_rpm = 30.0 / math.pi

# radius_m = 0.037

# romega = 2000.0 * rpm_to_rad_per_sec * radius_m
# v0_mps: float = 36.8611
# v_rel = v0_mps
# S = (romega / v_rel) * 1.0 #math.exp(-t / (self.tau * v_ref_ms / v_rel)) if v_rel > 0 else 0
# cl = cl2 * S / (cl0 + cl1 * S) if (cl0 + cl1 * S) > 0 else 0
# print(cl)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 日本語フォント設定（軸・タイトル等の文字化け対策）
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "IPAexGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def bumpy_terrain(x, y):
    """
    山と谷がたくさんある勾配（最適化の探索イメージ用）
    複数の sin/cos の重ね合わせで起伏を作る
    """
    z = (
        np.sin(x) * np.cos(y)
        + 0.5 * np.sin(2 * x) * np.cos(2 * y)
        + 0.3 * np.sin(3 * x + 1) * np.cos(2 * y - 0.5)
        + 0.2 * np.sin(x + y) * np.cos(x - y)
    )
    return z


def main():
    # メッシュ作成
    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    Z = bumpy_terrain(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 曲面を描画（カラーマップで高低を色分け）
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="terrain",
        edgecolor="none",
        alpha=0.9,
    )

    # Y = y_slice での XZ 断面：断面平面と曲面の交線を太線で強調
    y_slice = 0.5
    x_line = np.linspace(-4, 4, 200)
    z_line = bumpy_terrain(x_line, y_slice)

    # 断面平面（Y = y_slice の XZ 平面を半透明で表示）
    xp, zp = np.meshgrid(np.linspace(-4, 4, 2), np.linspace(Z.min(), Z.max(), 2))
    ax.plot_surface(xp, np.full_like(xp, y_slice), zp, alpha=0.2, color="blue")

    # 交線（太線で強調）
    ax.plot(x_line, np.full_like(x_line, y_slice), z_line, "r-", lw=6, label=f"Y = {y_slice} 断面の交線")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (目的関数)")
    ax.set_title("最適化")

    fig.colorbar(surf, ax=ax, shrink=0.6, label="Z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()