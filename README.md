# TD3 on CarRacing-v2

本專案是國立陽明交通大學（NYCU）**Selected Topics in Reinforcement Learning（強化學習專題）** 課程 **Lab 4：Twin Delayed DDPG (TD3)** 的作業實作。作業由 NYCU CGI Lab 規劃，講義與 sample code 由課程教授及助教提供，在此致謝。

> 本 repo 僅作為個人學習紀錄。若你正在修習這門課，請遵守課程的學術誠信規定，自行完成作業。

## 作業內容

用 TD3 從畫面像素學會駕駛 Gymnasium 的 CarRacing-v2，並做以下 ablation：

1. Twin Q-networks 與 single Q-network 的比較
2. 開啟與關閉 target policy smoothing
3. 不同的 delayed policy update 步數
4. 不同大小的 exploration noise
5. 自行設計比原始 reward 更好的 reward function

## 原始碼來源

| 部分 | 來源 |
|---|---|
| 網路架構（`models/`）、replay buffer、環境 wrapper 與影像前處理、訓練與評估框架 | 課程提供的 sample code |
| `### TODO ###` 標記的部分：選動作、critic 與 actor 更新（clipped double Q、target policy smoothing、delayed policy update）、soft target update | 作業實作 |
| Ablation 參數開關（`main.py`）、自訂 reward（`RewardShapingWrapper`）、`test.py`、`run_all.bat`、`package.py` | 作業延伸 |

## 結果

- 最終模型：TD3，replay buffer 50000、煞車維度的探索 noise × 0.1，訓練 320k 步。
- 在 60 條測試賽道（seed 0–59）上平均 **804.8**，其中 53 條 ≥ 800。
- 詳細的曲線、ablation 與討論見 [report.pdf](report.pdf)（精簡版）與 [report_full.pdf](report_full.pdf)（完整版），實驗紀錄見 [experiment_results.md](experiment_results.md)。

本 repo 不包含模型權重與 TensorBoard 紀錄。

## 環境

- Python 3.11
- gymnasium 0.29.1（CarRacing-v2）、torch 2.0.1、numpy 1.25.2、box2d-py 2.3.5，完整清單見 [src/requirements.txt](src/requirements.txt)
- Windows 安裝 box2d-py 需要 Visual Studio Build Tools（C++）與 swig，詳見 requirements.txt 開頭的說明

## 使用方式

```bash
cd src
# 訓練（預設設定，紀錄與 checkpoint 存到 ../runs/）
python main.py --total_steps 400000 --seed 0
# 最終模型的設定
python main.py --brake_noise_scale 0.1 --buffer_size 50000 --total_steps 500000 --seed 0
# Ablation 開關
python main.py --single_q                 # single Q-network
python main.py --no_target_smoothing      # 關閉 target policy smoothing
python main.py --policy_delay 4           # delayed update 步數
python main.py --expl_noise 0.3           # exploration noise
python main.py --reward_shaping grass     # 自訂 reward
# 測試（每個 seed 一條賽道，可錄影）
python test.py --checkpoint <model.pth> --seeds 0 1 2 3 4 --record_video ../videos
```

## 參考資料

- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML.
- Lillicrap, T. P. et al. (2015). Continuous control with deep reinforcement learning. arXiv:1509.02971.
- Silver, D. et al. (2014). Deterministic Policy Gradient Algorithms. ICML.
