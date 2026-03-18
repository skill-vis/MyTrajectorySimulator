# Rapsodo → Nathan Excel 変換

## ファイル

| ファイル | 内容 |
|----------|------|
| `clock_time_to_angle_deg.py` | Rapsodo 時計表記 → 角度 [deg] |
| `pitch_parameters_bsg.py` | `PitchParameters` + `angular_velocity_xyz_to_backspin_sidespin_wg` |
| `rapsodo_to_nathan.py` | Rapsodo 入力 → Nathan Excel 用 `PitchParameters` / 1行出力 |

## セットアップ

```bash
cd code
pip install -r requirements.txt
python rapsodo_to_nathan.py
```

