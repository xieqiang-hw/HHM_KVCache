# HHM–ASU KV Cache 仿真实验分析

结果快照采集于 2026-11-25 09:11:30（北京时间）。10,000 条组完成 4 组，全量组完成 2/5 组。
[打开交互报告](实验结果分析.html) · [全量日志](hhm_kvcache_sim/results/balanced_routing/full_607190_batch.log)

## 主要发现

- **全量回放。** 64× 和 128× 的 HHM 命中率分别为 96.58% 和 94.59%，三层合计分别为 96.95% 和 96.48%。TTFT P90 分别为 2.73 和 5.82 秒。
- **全量写入。** 64× 累计写入 WIN 4,179.36 GB、ASU 1,669.74 GB。128× 累计写入 WIN 8,227.83 GB、ASU 4,118.74 GB。
- **10,000 条组。** 三层合计命中率为 95.82%–96.87%。TTFT P90 为 1.25–1.69 秒。

## 回放窗口与缓存命中

| 实验 / 请求数 | 窗口跨度 h | 窗口密度 | DRAM | WIN | HHM 合计 | ASU 补命中 | 三层合计 | 理论基线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1× / 10,000 | 625.29 | 0.15× | 95.82% | 0.00% | 95.82% | 0.00% | 95.82% | 97.07% |
| 16× / 10,000 | 8.51 | 10.76× | 94.64% | 1.93% | 96.57% | 0.49% | 96.59% | 97.34% |
| 32× / 10,000 | 4.15 | 22.09× | 95.64% | 1.22% | 96.86% | 0.47% | 96.87% | 97.68% |
| 64× / 10,000 | 2.25 | 40.71× | 95.67% | 0.90% | 96.57% | 0.33% | 96.58% | 97.62% |
| 64× / 607,190 | 794.75 | 7.00× | 86.84% | 9.74% | 96.58% | 10.97% | 96.95% | 98.23% |
| 128× / 607,190 | 751.29 | 7.40× | 76.51% | 18.08% | 94.59% | 35.00% | 96.48% | 98.23% |

## 写入量与淘汰

| 实验 / 请求数 | WIN 写入 GB | ASU 写入 GB | 单卡 WIN 峰值 GB | WIN 淘汰块 | ASU 淘汰块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1× / 10,000 | 1.76 | 0.10 | 1.76 | 0 | 0 |
| 16× / 10,000 | 13.46 | 2.25 | 7.76 | 0 | 0 |
| 32× / 10,000 | 13.68 | 2.29 | 5.80 | 0 | 0 |
| 64× / 10,000 | 5.67 | 0.68 | 5.29 | 0 | 0 |
| 64× / 607,190 | 4,179.36 | 1,669.74 | 91.14 | 0 | 0 |
| 128× / 607,190 | 8,227.83 | 4,118.74 | 169.48 | 0 | 0 |

## 时延与调度

| 实验 / 请求数 | TTFT P90 s | TPOT P90 ms | 端到端 P90 s | 平均 Decode BS | 活跃 Decode rank | 待准入 token 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1× / 10,000 | 1.56 | 11.47 | 8.34 | 1.02 | 64 | 171,414 |
| 16× / 10,000 | 1.34 | 11.78 | 7.61 | 1.03 | 64 | 418,546 |
| 32× / 10,000 | 1.25 | 11.84 | 7.11 | 1.04 | 64 | 461,380 |
| 64× / 10,000 | 1.69 | 12.52 | 8.16 | 1.05 | 64 | 489,645 |
| 64× / 607,190 | 2.73 | 13.61 | 10.37 | 1.07 | 64 | 2,069,011 |
| 128× / 607,190 | 5.82 | 32.79 | 17.30 | 1.16 | 64 | 2,727,534 |

## 未完成实验

| 实验 / 请求数 | 状态 | 已完成请求 | 进度 | 累计运行 |
| --- | ---: | ---: | ---: | ---: |
| 256× / 607,190 | 失败（MemoryError） | 215,973 | 35.57% | 02:01:05 |
| 512× / 607,190 | 已停止 | 53,846 | 8.87% | 12:57:04 |
| 1024× / 607,190 | 已停止 | 41,878 | 6.90% | 12:56:53 |

## 统计口径

- 完整 trace 有 607,190 条请求。1× 按原始顺序取样，其他倍率先叠加再按到达时间取样。路由按预计 TTFT 选择 DP rank。
- 配置为 8 台机器 × 8 个 DP rank，每卡 DRAM 24 GB、WIN 可用 211 GB，ASU 为 12 × 64 TB，KV 块为 16 token。
- 窗口跨度取所选请求的首末到达时间，窗口密度倍率为所选窗口请求密度除以完整原始 trace 的平均请求密度。
- HHM 命中包含 DRAM 和 WIN。三层合计按全部输入 token 计算。ASU 补命中按 HHM 未命中的输入 token 计算。理论基线使用仿真输出的 `ideal`。
- 写入和容量用十进制 GB。Decode BS 按各 rank 的活跃 decode step 加权。待准入 token 峰值包含后续可能命中的前缀。

## 结果文件

- 1× / 10,000 · [汇总](hhm_kvcache_sim/results/balanced_routing/1x_10000/summary.json) · [时间序列 CSV](hhm_kvcache_sim/results/balanced_routing/1x_10000/timeseries.csv) · [六联图](hhm_kvcache_sim/results/balanced_routing/1x_10000/timeseries.svg) · [运行日志](hhm_kvcache_sim/results/balanced_routing/1x_10000/run.log)
- 16× / 10,000 · [汇总](hhm_kvcache_sim/results/balanced_routing/16x_10000/summary.json) · [时间序列 CSV](hhm_kvcache_sim/results/balanced_routing/16x_10000/timeseries.csv) · [六联图](hhm_kvcache_sim/results/balanced_routing/16x_10000/timeseries.svg) · [运行日志](hhm_kvcache_sim/results/balanced_routing/16x_10000/run.log)
- 32× / 10,000 · [汇总](hhm_kvcache_sim/results/balanced_routing/32x_10000/summary.json) · [时间序列 CSV](hhm_kvcache_sim/results/balanced_routing/32x_10000/timeseries.csv) · [六联图](hhm_kvcache_sim/results/balanced_routing/32x_10000/timeseries.svg) · [运行日志](hhm_kvcache_sim/results/balanced_routing/32x_10000/run.log)
- 64× / 10,000 · [汇总](hhm_kvcache_sim/results/balanced_routing/64x_10000/summary.json) · [时间序列 CSV](hhm_kvcache_sim/results/balanced_routing/64x_10000/timeseries.csv) · [六联图](hhm_kvcache_sim/results/balanced_routing/64x_10000/timeseries.svg) · [运行日志](hhm_kvcache_sim/results/balanced_routing/64x_10000/run.log)
- 64× / 607,190 · [汇总](hhm_kvcache_sim/results/balanced_routing/64x_607190/summary.json) · [时间序列 CSV](hhm_kvcache_sim/results/balanced_routing/64x_607190/timeseries.csv) · [六联图](hhm_kvcache_sim/results/balanced_routing/64x_607190/timeseries.svg) · [运行日志](hhm_kvcache_sim/results/balanced_routing/64x_607190/run.log)
- 128× / 607,190 · [汇总](hhm_kvcache_sim/results/balanced_routing/128x_607190/summary.json) · [时间序列 CSV](hhm_kvcache_sim/results/balanced_routing/128x_607190/timeseries.csv) · [六联图](hhm_kvcache_sim/results/balanced_routing/128x_607190/timeseries.svg) · [运行日志](hhm_kvcache_sim/results/balanced_routing/128x_607190/run.log)
