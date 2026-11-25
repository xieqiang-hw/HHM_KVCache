# HHM–ASU KV Cache 仿真

Python 3.10+，只依赖标准库。默认读取项目 `data/claude-code-main-glm52-128k-exact-sample` 中从 WSL 截取的前 256 条连续请求。样本保留了原始毫秒时间戳、输入和输出 token ID，来源记录在样本的 `manifest.json` 中。命令从本目录运行。

```bash
python run.py --output results/sample
python run.py --config stress.json --output results/stress
python -m unittest -v
```

第一条使用设计规格和原始到达间隔。第二条缩小缓存、减少机器和 rank、压缩到达间隔，用来观察回收路径，不能用它的时延和生产配置直接比较。参数可在 JSON 中覆盖，完整字段和单位见 `workload.py` 的 `Config`。对超过 50 MB 的本地样本，可用 `--limit 40` 快速试跑。

直接在 WSL 中运行时，可将 `--trace` 指向原始 exact 目录或其中的 `requests.jsonl`，用 `--skip`、`--limit` 选择窗口。全量文件约 178 GB，建议先用窗口。程序逐行解析请求，结果中的请求时延样本、理想命中集合及仍在驻留的块会随运行规模占用内存。可用 `--no-batch-csv` 关闭逐步 CSV。重新抽样可用 `python sample_trace.py SOURCE_DIRECTORY NEW_SAMPLE_DIRECTORY --count 256 --skip 0`。

## 全量精简 trace

`compact_trace.py` 逐段读入 exact JSONL，将实际的输入输出长度、原始毫秒时间戳和预计算的完整 KV 块 Key 写成 gzip 分片。同一天可有多个分片。相邻请求共有的 Key 前缀只存一次，读取时恢复完整 Key 列表。该格式保留仿真所需信息，原始 token ID 仍在 exact 源文件中。

```bash
python compact_trace.py EXACT_SOURCE_DIRECTORY ../data/claude-code-main-glm52-128k-compact --workers 12 --chunk-mib 512
python run.py --trace ../data/claude-code-main-glm52-128k-compact --limit 320 --output results/compact
```

转换结束后才生成顶层 `manifest.json`，仿真从这份清单按顺序读取分片。若转换中断，使用同一源目录、目标目录、分段大小和压缩级别重跑即可续传。精简目录支持 `--skip` 和 `--limit`，可直接跳过窗口之前的分片。

## 会话叠加

```bash
python3 -u run.py --trace ../data/claude-code-main-glm52-128k-compact \
  --overlay 8 --output results/overlay-8x --no-batch-csv
```

默认沿用 `prepare_swe_chat_overlay.py` 的口径。仿真把原时间轴等分为指定段数，一个会话的全部请求归它首条请求所在的段。将**各段起点**对齐到模拟时间 0，再按平移后的时间归并请求。段起点到首条请求之间的空窗及跨段会话尾部都保留。把 `--overlay 8` 改成 `--overlay 16` 或 `--overlay 32` 即可切换倍率。若要改为各段首条请求都在时间 0 到达，可以显式指定 `--overlay-align first-request`。

该模式直接读取已有的压缩分片。会话索引 `sessions.idx` 是从原始 `source_requests.jsonl` 生成的小文件，只包含原始时间戳和匿名会话编号。若目录缺少索引，用 `python3 overlap.py COMPACT_DIRECTORY ORIGINAL_SOURCE_REQUESTS_JSONL` 生成一次。也可通过 `--session-index` 指定索引路径。

叠加使用完整 trace 确定分段和会话归属，再按叠加后的到达时间排序。同一时刻按原始行号排序。`--skip` 跳过回放顺序的前几条，`--limit` 取接下来的若干条，保留叠加后的时间戳。比如 `--overlay 128 --limit 20000` 回放完整 128× 叠加后的前 20,000 条。若另设 `--time-scale`，则在选取后缩放时间。默认不设 `--overlay` 时，`--skip` 和 `--limit` 仍按原始 trace 顺序生效。

`summary.json` 的 `overlay.selection_order` 为 `after-overlay`。`effective_density_multiplier` 用完整原始跨度除以完整叠加跨度，`selected_density_multiplier` 则用所选窗口的请求密度除以原始完整 trace 的请求密度。`selected_*` 记录选取范围，活跃分段、会话数和长尾请求数也按所选请求统计。此前先截取原始 trace 再叠加的结果使用旧口径。

## 代码怎么读

- `workload.py` 读 trace，并用「前一块的前缀哈希 + 当前块真实 token」为完整块生成 Key。精确 token 允许将既往 decode 的完整 KV 块用于后续前缀命中。尚未形成完整块的尾部占本卡空间，最后一个刚输出、尚未被模型消费的 token 没有 KV。
- `simulator.py` 按请求到达和机器 forward 完成推进事件。默认用 `estimated_ttft` 路由，比较机器 forward 等待、rank 的准入积压、剩余 prefill 与暴露的 ASU 读取时间。准入等待是估算，实际入批仍使用缓存容量和序列约束。配置 `"routing_policy": "cache_first"` 可回放旧策略。rank 独立组批，一台机器以最慢 rank 的计算时延作为同步边界。每步优先 prefill，再用剩余预算推进 decode。新生成的 KV 在 forward 完成后可复用。
- `cache.py` 管理本卡前缀树、DRAM/WIN 驻留前沿和全局 ASU。DRAM 优先释放重复副本，其次将稳定热块迁至 WIN，最后从前沿选择块备份或丢弃。WIN 只回收冷却且住满的前沿块。备份逐个检查路径并保持已存在的祖先锁定，缺失的完整块先预留 ASU 空间，整个备份计划通过后再安排传输。新 ASU 副本在 forward 完成时按前缀顺序发布；跨卡缓存复用只经 ASU。

## 仿真口径

- 默认 8 机 × 8 rank，DP8EP8，24 GB DRAM、211 GB 可用 WIN、12 × 64 TB ASU。每 rank 共用 64 序列和 32,768 token 的单步预算。前缀命中按完整块计数；完整输入均命中时保留一步最后 logits 的计算。MTP5 的默认输出概率为 `[0.05, 0.10, 0.20, 0.25, 0.25, 0.15]`，均值为 4；入批按最多 6 个验证 token（受剩余输出长度限制）占预算，实际生成量由分布抽样，并以 trace 的输出长度为止。
- 新产生和从 ASU 读回的 KV 进入 DRAM。请求入批时为预测会用到的 DRAM 块先预留空间，避免活跃请求填满后无法继续 decode。WIN 命中保持本卡驻留并记录滚动读取量。仅在 DRAM 空间不足时，已有缓存块才按迁移规则进入 WIN。
- `tau_s`、冷热阈值、WIN 驻留期采用设计图中的示意值；每步每 rank 的主动搬运预算默认 50 ms，按名义服务时间扣减。ASU 用全局逻辑 LRU，按 Key 映射到设备带宽通道，设备流量按已规划顺序排队。算传重叠用 `transfer_overlap` 对迁移和加载折算，实际 forward 结果等传输完成后才提交。WIN 滚动读视为完全掩盖。仿真在 forward 边界抽象本地回收和空间计划，不模拟算子、逐层事件、ASU IOPS 或单独的控制消息时延。
- 普通备份按复用分数触发。若在测试 trace 中显式给记录填 `share_prefix_blocks`，它表示已知共享需求，可在本次 prefill 完成时绕过目标块的分数门槛，仍须符合完整路径、容量和预算约束。真实样本未携带这个字段。空的本地树分支会清理，仍驻留或连接到驻留后继的节点保留复用分数。
- 理想命中率用无限容量、此前到达请求的完整上下文立即可复用作基线，忽略实际完成时间。HHM/ASU 命中率都按输入 token 加权，ASU 条件命中率的分母是所有未被本卡 HHM 连续命中覆盖的输入 token。TPOT 按请求的「末 token 时间减首 token 时间，再除以输出 token 数减一」统计。时间戳是 trace 的首响应事件代理，到达节奏按原始间隔回放。

`summary.json` 包含 DRAM/WIN/ASU 命中率、理想命中率、WIN/ASU 写入字节数、各 rank 平均 decode batch size 和 TTFT/TPOT/TE2E 的均值及 P90。`batches.csv` 按机器 forward 和 rank 记录时间、decode batch size 及 P/D 调度 token 数，空闲 rank 的 batch size 记作 0。

每次回放也生成 `timeseries.svg` 六联图和对应的 `timeseries.csv`。命中率和请求时延按请求到达窗口统计，ASU 曲线使用 HHM 未命中时的条件命中率。Decode BS 是各 rank 有 decode 工作的 forward 的平均批量。集群排队 Prefill token 是尚未入批的请求的输入 token 数之和，展示时间加权均值和峰值。TTFT、TPOT、TE2E 同时绘制均值和 P90。

运行时默认向标准错误流显示进度。百分比按「已完成请求 / 本次回放总请求」计算，总数来自 trace 清单并考虑 `--skip` 和 `--limit`。没有清单时显示已完成和已到达的请求数。交互终端原地刷新，重定向到日志时定期输出一行，可用 `--no-progress` 关闭。

时间轴从本次回放第一条请求起算，默认以 60 秒为基础窗口，可用 `--metrics-interval-s` 调整。长回放自动合并相邻窗口，使图和时序 CSV 不超过 240 个窗口。没有样本的命中率、Decode BS 和时延留空。使用 `--no-batch-csv` 时仍生成这两份时序文件。
