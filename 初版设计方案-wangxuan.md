我们发现该系统与现有系统有几个不一样的点：

HHM-DRAM 24GB 6.4TBps  HHM-WIN 320GB 800GBPs  12*ASU 每个ASU 64TB 40GBps

互联域不同：HHM是每卡独有，ASU是全局共享。这一点和hicache的HBM+DRAM+SSD最像，唯一的差别是DRAM这一层是机内共享的，而HHM-WIN则是卡内独享的。不过hicache的DRAM这一层本身也不会跨卡复用，所以软件上可能刚好一样了。

互联方式不同：仅仅允许 HHM-DRAM <-> HHM-WIN, HHM-DRAM <-> ASU, HHM-WIN <-> ASU 这三种南北向数据通路，数据不允许走P2P东西向通路（但是控制信号还是可以走的）。

ASU这一层其实是一种类似nvidia ICMS的高带宽、高IOPS、全局池化的KV语义存储层，可以抽象为具有5种基本操作：
     - lookup: 查询一个Key是否存在
     - lock/unlock: 要使用一个key之前，先锁住，避免它在使用时ASU内部的自动淘汰算法把它淘汰了。unlock则是解锁。
     - read: 指定key，读取value(也即kvcache block数据)，会直通到NPU HHM上
     - write/new: 指定key，写value，同样会直通。读支持部分读取，写必须整块写入
     - delete: 指定key，主动删除它，腾出空间（另外ASU内根据自身剩余容量，也会主动淘汰掉一些block，淘汰算法是LRU）
     - 以上操作也可能组合出一些合并的原子操作，比如 lookup+lock 原子操作
   需要注意区分 "KV语义" 和 "KVcache" ，KV语义是指由指定的Key，可以lookup/lock/unlock/read/write/delete特定的Value，只不过这个Value存的实际上是kvcache block。而kv语义的key可以是kvcache的token hash+position
此外，我们针对KVC管理系统还给出这几个设计原则：
1.算传同步：为了设计简单，所有的前缀缓存在数据面的传输（包括promote和demote）都必须和model.forward一起走，也即同步执行（可跨层掩盖），（比如layer1.forward的时候layer2.load是同时进行的，以此类推），但是不能异步执行。



●该仿真系统是要在一定抽象程度下，能够仿真多节点的前缀缓存行为。不需要把所有细节都仿真出来
目标模型：GLM-5.2-w8a8c8，每token KVcache是50kB。在DP8EP8下每卡权重占用HHM-WIN=780GB/8+12GB=109GB。剩下320GB-109GB都可以给KVcache用
 







仿真粒度：最细只到 model.forward 级别，假设我们知道 model.forward 的时间，然后对系统进行仿真，不需要仿真到 model.forward() 内部的算子

系统中全部进行PD共部，有多台机器，每个机器是一个EP域，内部有多个DP rank（比如说如果有4机，每机8卡DP8EP8，那么相当于有4个EP域，每个EP域有8个DP rank）
     - EP域之间的 model.forward 是异步的，EP域内不同的DP rank的 model.forward 是同步的，每个DP rank一次处理一个请求（prefill或者decode）

vllm推理行为：
   - 每台机器上面有个 vllm ，相对独立的进行任务调度。类似于vllm PD共部的scheduler，每进行完一上一个 model.forward ，就立即决策下一次 model.forward 该推什么。prefill和decode都是 DP8EP8
   -单节点vllm scheduler调度决策：
       - prefill和decode running请求占用HHM-DRAM或者HHM-WIN空间
       - 只要 HHM kv blocks 能够塞得下一个 prefill 请求，就优先调度 prefill
       - 当 prefill 进行时，当前节点如果有正在 decode 的请求，就进行continuous batching，此时相当于prefill和decode混合进行model.forward
       - 没有请求需要 prefill 时（比如显存不够塞得下一个新的 prefill ，或者已经没有prefill在排队了），则进行纯 decode (组 batch decode)
       - 会有一个 max num batched tokens 配置项, 一次 model forward 推的总 token 数不能超过该值 .  max num batched tokens 默认为 32768
       - decode 开启 MTP5 ，一次 model.forward 一个吐出 1~6 个 token ，可配置输出各个token数量 的概率。平均应该时吐4 token


model.forward 时延如何模拟？
       - 对于纯 prefill step 和 prefill+decode混合step， model.forward time = num batched tokens / prefill TPS, 其中 prefill TPS 是一个可配置的固定值，默认取 2000 TPS per NPU 。???
       -  prefill时，机器配置为 TP8EP8 ，也即整个机器的所有卡都在推一个 prefill 请求。然后 kvcache则可以落在该机器的任意一个卡上，后续就在这个卡（DP rank）上进行 decode
       - 对于纯 decode step , model.forward time (ITL) 只与 decode batch-size 有关，见 :
            | 每卡 batch-size | ITL (step time) /ms |
            | --------------- | ------------------- |
            | 1               | 40.6                |
            | 4               | 44.5                |
            | 8               | 53.2                |
            | 12              | 59.1                |
            | 16              | 66.7                |
            | 20              | 73.1                |
            | 24              | 83.2                |
            | 28              | 88.6                |
            | 32              | 97.9                |
            | 60              | ？      |
            | 80              | ？     |
            最大只支持 decode batchsize=80，不允许更大。该表格中缺失的点（比如bs=2,3,5,6,7）需要你按照这个规律拟合一个出来
            decode 为 DP8EP8 ，不同 DP rank 推不同的请求。
 
系统配置：MTP5 Ascend950DT, GLM-5.3-W8A8C8, 纯decode, 每种序列长度下尽量打大batch把显存打满

序列长度 每卡bs TPOT_mean/ms TPOT_p90 每卡吞吐/TPS
DEP32 4096 80 50.3 60.4 1305
DEP32 16384 37 29.9 33.6 1104
DEP32 65536 12.5 14.6 16.1 716
DEP32 131072 8 16.2 19.9 335
DEP16 4096 60 38.8 52.4 1439
DEP16 16384 20 17.5 19.7 1001
DEP16 65536 5 13.6 16.1 320
DEP16 131072 2 11.4 11.7 152



running请求（正在prefill或者decode）的KVcache：
    - 如果命中HHM-WIN，不需要加载到HHM-DRAM
    - 如果命中ASU，加载到HHM-DRAM才能推

集群调度器行为：全局有个任务调度器，用来接收客户端发来的推理请求，并分配给每个vllm（会分到具体的DP rank上）。分配任务要看2个原则
   - 原则1(更重要)：尽量分配到 HHM 命中率高的那台机器上，在这台机器内，则尽量分配到L1命中率高的那个DP rank上。
   - 原则2：还要看这个机器/DPrank的任务排队情况，如果一个节点负载比较轻（也即正在排队，还未进行prefill的序列比较少），则优先调度到负载轻的节点和DP rank上。
   - 具体的调度策略则是以上2个原则的混合。
  -->

!-- 8 评价指标：


命中率原则：
     - HHM命中率越高越好。在HHM内部，命中WIN和命中DRAM是一样的（因为WIN命中可以通过滚动加载消除其影响，对性能没有任何影响）
     - ASU命中率越高越好。如果一个块没有命中 HHM ，就看 ASU有没有命中。ASU命中相比HHM命中有特定的性能损失，但总比完全没命中要好
     - 仿真的时候需要加一个基线系统：理论最高命中率（即假设除新输入token外其他token均可复用时的命中率）

降写原则：HHM-WIN和ASU都有特定的写寿命，应该尽量少写。其中HHM-WIN的寿命问题更严重一些。

decode平均Batchsize以及batchsize随着时间的变化

请求的mean和P90的TTFT, TPOT, TE2E。



KVcache管理算法https://github.com/xwLearnsLLM/PrefixCacheSurvey/blob/main/HHM_ASU_native_KVC_management.md