# MyBallTrajectorySim 計算式

野球ボール軌道シミュレータ（MyBallTrajectorySim.py）で使用される式を数学記法でまとめる。

---

## 1. 空気密度

**飽和水蒸気圧（Buck式）**

$$
p_{\mathrm{sat}} = 4.5841 \exp\left(\frac{(18.687 - T_C/234.5) \cdot T_C}{257.14 + T_C}\right) \quad [\mathrm{mmHg}]
$$

**空気密度**

$$
\rho = 1.2929 \cdot \frac{273}{T_C + 273} \cdot \frac{p_0 \cdot e^{-\beta h} - 0.3783 \cdot \frac{RH}{100} \cdot p_{\mathrm{sat}}}{760} \quad [\mathrm{kg/m}^3]
$$

（$T_C$: 気温 [°C], $h$: 標高 [m], $RH$: 相対湿度 [%], $p_0$: 海面気圧 [mmHg], $\beta = 0.0001217\,\mathrm{m}^{-1}$）

---

## 2. 抗力・マグヌス係数用定数

**断面積**

$$
A = \pi r^2
$$

**定数 $c_0$（抗力加速度の係数）**

$$
c_0 = \frac{1}{2} \frac{\rho A}{m}
$$

---

## 3. 抗力係数

$$
C_d = C_{d0} + C_{d,\mathrm{spin}} \cdot \frac{\omega_{\mathrm{eff}}}{1000}
$$

（$C_{d0} = 0.297$, $C_{d,\mathrm{spin}} = 0.0292$, $\omega_{\mathrm{eff}}$: 有効スピン [rpm]）

---

## 4. 揚力係数（マグヌス用）

**回転パラメータ $Q$**

$$
Q = \frac{r\omega}{v_{\mathrm{rel}}}
$$

**揚力係数**

$$
C_L = \frac{C_{L2} \cdot Q}{C_{L0} + C_{L1} \cdot Q}
$$

（$C_{L0} = 0.583$, $C_{L1} = 2.333$, $C_{L2} = 1.12$）

---

## 5. 有効スピン

$$
\omega_{\mathrm{eff}} = \sqrt{\omega_{\mathrm{total}}^2 - \left(\frac{30}{\pi}\right)^2 \left(\frac{\boldsymbol{\omega} \cdot \mathbf{v}}{v}\right)^2} \quad [\mathrm{rpm}]
$$

**回転速度（マグヌス計算用）**

$$
r\omega = \omega_{\mathrm{eff}} \cdot \frac{\pi}{30} \cdot r \quad [\mathrm{m/s}]
$$

---

## 6. 相対速度（風込み）

$$
\mathbf{v}_{\mathrm{rel}} = \mathbf{v} - \mathbf{v}_{\mathrm{wind}} \quad (z \geq h_{\mathrm{wind}} \text{ のとき})
$$

**風速成分**

$$
v_{x,w} = v_{\mathrm{wind}} \cdot 0.44704 \cdot \sin\phi_{\mathrm{wind}}, \quad v_{y,w} = v_{\mathrm{wind}} \cdot 0.44704 \cdot \cos\phi_{\mathrm{wind}}
$$

---

## 7. 抗力（空気抵抗）の加速度

$$
a_{\mathrm{drag},x} = -c_0 \cdot C_d \cdot v_{\mathrm{rel}} \cdot (v_x - v_{x,w})
$$

$$
a_{\mathrm{drag},y} = -c_0 \cdot C_d \cdot v_{\mathrm{rel}} \cdot (v_y - v_{y,w})
$$

$$
a_{\mathrm{drag},z} = -c_0 \cdot C_d \cdot v_{\mathrm{rel}} \cdot v_z
$$

---

## 8. マグヌス効果の加速度

**ベクトル表記（角速度と相対速度の外積）**

$X = r\omega / (r\omega_0)$ とおく：

$$
\mathbf{a}_{\mathrm{Magnus}} = c_0 \, \frac{C_L}{\omega_{\mathrm{total}}} \, \frac{v_{\mathrm{rel}}}{X} \, \left( \boldsymbol{\omega} \times \mathbf{v}_{\mathrm{rel}} \right)
$$

マグヌス力は角速度ベクトルと速度ベクトルの外積に比例する：

$$
\mathbf{a}_{\mathrm{Magnus}} \propto \boldsymbol{\omega} \times \mathbf{v}_{\mathrm{rel}}
$$

**成分表示**

$$
a_{M,x} = c_0 \cdot \frac{C_L}{\omega_{\mathrm{total}}} \cdot \frac{v_{\mathrm{rel}}}{X} \cdot (\omega_y v_z - \omega_z v_y)
$$

$$
a_{M,y} = c_0 \cdot \frac{C_L}{\omega_{\mathrm{total}}} \cdot \frac{v_{\mathrm{rel}}}{X} \cdot (\omega_z v_x - \omega_x v_z)
$$

$$
a_{M,z} = c_0 \cdot \frac{C_L}{\omega_{\mathrm{total}}} \cdot \frac{v_{\mathrm{rel}}}{X} \cdot (\omega_x v_y - \omega_y v_x)
$$

---

## 9. 運動方程式（合成加速度）

$$
\frac{d\mathbf{v}}{dt} = \mathbf{a}_{\mathrm{drag}} + \mathbf{a}_{\mathrm{Magnus}} + \mathbf{g}
$$

$$
\mathbf{g} = (0, 0, -g), \quad g = 9.79 \;\mathrm{m/s}^2
$$

---

## 10. 初期速度（リリース角度・方向）

$\theta$: リリース角度 [deg]（正＝下向き）、$\phi$: リリース方向 [deg]

$$
v_x = v_0 \cos\theta \sin\phi
$$

$$
v_y = -v_0 \cos\theta \cos\phi
$$

$$
v_z = -v_0 \sin\theta
$$

---

## 11. 数値積分（補足）

**RK4（4次ルンゲ・クッタ）**

$$
\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

**オイラー法**

$$
\mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t, \quad \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t + \frac{1}{2}\mathbf{a}_n (\Delta t)^2
$$

---

## 12. 角速度 (wx, wy, wz) と BSG (backspin, sidespin, wg) の変換

$B$: backspin [rpm], $S$: sidespin [rpm], $G$: wg（ジャイロ）[rpm]。  
$\theta$: リリース角度 [rad], $\phi$: リリース方向 [rad]。  
$c_\theta = \cos\theta$, $s_\theta = \sin\theta$, $c_\phi = \cos\phi$, $s_\phi = \sin\phi$ とする。

**初速度の単位ベクトル（進行方向）**

$$
\hat{\mathbf{u}} = \frac{\mathbf{v}_0}{v_0} = (c_\theta s_\phi,\; -c_\theta c_\phi,\; -s_\theta)^{\top}
$$

**3 軸の単位ベクトル**

- バックスピン軸（水平・速度の水平成分に垂直）: $\hat{\mathbf{b}} = (-c_\phi,\; s_\phi,\; 0)^{\top}$
- サイドスピン軸: $\hat{\mathbf{s}} = (-s_\theta s_\phi,\; -s_\theta c_\phi,\; c_\theta)^{\top}$
- ジャイロ軸（速度方向）: $\hat{\mathbf{g}} = \hat{\mathbf{u}}$

**正変換（BSG → 角速度 [rad/s]）**

$k = \pi/30$（rpm → rad/s）とおくと、

$$
\boldsymbol{\omega} = k\,\bigl( B\,\hat{\mathbf{b}} + S\,\hat{\mathbf{s}} + G\,\hat{\mathbf{g}} \bigr)
$$

成分で書くと、

$$
\begin{pmatrix} \omega_x \\ \omega_y \\ \omega_z \end{pmatrix}
= k\,
\begin{pmatrix}
-c_\phi & -s_\theta s_\phi & c_\theta s_\phi \\
s_\phi  & -s_\theta c_\phi & -c_\theta c_\phi \\
0       & c_\theta         & -s_\theta
\end{pmatrix}
\begin{pmatrix} B \\ S \\ G \end{pmatrix}
= k\,\mathbf{M}
\begin{pmatrix} B \\ S \\ G \end{pmatrix}
$$

**逆変換（角速度 [rad/s] → BSG [rpm]）**

$$
\begin{pmatrix} B \\ S \\ G \end{pmatrix}
= \frac{30}{\pi}\,\mathbf{M}^{-1}
\begin{pmatrix} \omega_x \\ \omega_y \\ \omega_z \end{pmatrix}
$$

実装では $\mathbf{M}^{-1}$ を数値計算（`np.linalg.inv(M)`）で求め、上式で $B,S,G$ を算出している。
