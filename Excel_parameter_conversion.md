# Excel「TrajectoryCalculator-new-3D-May2021.xlsx」との同一条件での計算について

同じ条件で Excel と Python（run_sim_and_visualize.py / MyBallTrajectorySim.py）の両方で計算することは**可能**です。  
単位が違うため、Excel には**換算した値**を入力してください。

## 単位対応（MKS → Excel で使う単位）

| 項目 | Python (MKS) | Excel (ヤード・ポンド) | 換算 |
|------|--------------|------------------------|------|
| 位置 x, y, z | m | ft | 1 m = 3.28084 ft |
| 初速度 v0 | m/s | mph | 1 m/s = 2.23694 mph |
| 角度 theta, phi | deg | deg | そのまま |
| 回転数 backspin, sidespin, wg | rpm | rpm | そのまま |
| 気温 | °F (temp_F) | °F | そのまま |
| 標高 | m (elev_m) | ft | 1 m = 3.28084 ft |
| 相対湿度 | % | % | そのまま |
| 気圧 | inHg | inHg | そのまま |
| 風速 | mph | mph | そのまま |
| 風向 | deg | deg | そのまま |
| 風の高さ | m (hwind_m) | ft | 1 m = 3.28084 ft |

## 手順

1. **Python で実行**  
   `python run_sim_and_visualize.py` を実行すると、ターミナルに  
   **「Excel (TrajectoryCalculator-new-3D-May2021.xlsx) 用パラメータ」** が表示されます。

2. **表示された数値を Excel にコピー**  
   表示されている ft, mph の値を、Excel の対応する入力セルにそのまま入力します。

3. **角度の符号**  
   Excel も「負の theta = 下向き」の慣習であれば、theta_deg はそのまま使えます。  
   逆だった場合は、Excel には `-theta_deg` を入力してください。

## 注意

- Excel のシートによってセル位置や名前が異なります。上記の「項目名」とシートのラベルを照らして入力してください。
- 計算式（Cd の扱いや const など）は Excel 版と MyBallTrajectorySim で一部差がある場合があり、結果が完全に一致しないことがあります。
