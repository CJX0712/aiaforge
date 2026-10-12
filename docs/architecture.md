# AiaForge · 架构文档

> 世界顶级属性推断攻击（Attribute Inference Attack）系统。零重写·集大成·闭环交付·可验证。
> 作者：晨星 (CJX0712)

## 1. 问题定义（威胁模型）

给定一个在「敏感属性 `S` 参与训练」的分类模型 `f: [X_pub, S] → Y`，攻击者在**推理时只知道公开特征 `X_pub` 与标签 `Y`、不知道 `S`**，通过向模型喂入两种候选 `S_guess ∈ {0,1}` 并观察输出，推断目标个体的真实 `S`。

这是与成员推断（MIA）并列但任务不同的隐私攻击：
- MIA 回答「该样本是否在训练集中」；
- AIA 回答「该样本的敏感属性 `S` 是什么」。

文献锚点：Yeom et al. 2020 *Privacy Risk of Overfitting*（置信阈值属性推断）、Wu et al. 2022 *ML-Doctor*（属性推断作为标准诊断项）、Fredrikson 2015 *Model Inversion*（梯度/置信反演）。

## 2. 系统骨架（单向无环）

```
cli.py ──▶ pipeline/ ──▶ { data/, aia/target, aia/methods, eval/ } ──▶ core/
                                       │
                                  aia/_numpy_models  (离线兜底)
```

| 层 | 职责 | 关键契约 |
|----|------|----------|
| `core/` | types / errors(E100~E500) / config(ENV 覆盖+校验) / interfaces(Protocol) / seed(全局确定性) | `set_all(seed)` 钉 BLAS 线程=1，保证逐位复现 |
| `data/` | 合成 DGP + 外部数组装载 | 生成器固定 seed；train/aux/test **连续不重叠** |
| `aia/target` | 被测目标模型（sklearn MLP / LogReg + NumPy 兜底） | 始终以 `[X_pub, S]` 为输入训练 |
| `aia/methods` | 攻击方法 + 旗舰 `AiaFuse` | `scores = P(S=1 | evidence)`，量纲统一 [0,1] |
| `eval/` | AUC / Accuracy / 显著性门禁 | AUC 单调不变，离线手工实现兜底 |
| `pipeline/` | `run_benchmark` + `Pipeline.run()` | 三分区严格隔离，无泄漏 |

## 3. 攻击方法清单（接口统一：分越大越像 S=1）

| 方法 | 类型 | 说明 |
|------|------|------|
| `majority` | 基线 | 永远预测辅助集多数类 `P(S=1)` |
| `label_only` | 基线 | 仅凭已知标签 `Y` 推断 `S`（经验 LR） |
| `conf_inversion` | 经典 | Yeom 式置信反演：喂 `S=0/1` 取对真标签置信更高者，gap→sigmoid |
| `attack_lr` | 元分类器 | 在「攻击者可见特征」`(X_pub, Y, P| S=0, P| S=1, gap)` 上训练 LR |
| `attack_nn` | 元分类器 | 同特征上的 MLP 元分类器（sklearn；缺失时退化为 NumPy LR） |
| **`aiafuse`** | **旗舰** | 校准堆叠融合上述组件 + **非劣守卫**（退化则回退最佳单法）+ 场景路由（无 `X_pub` 时退 `label_only`） |

## 4. 旗舰 AiaFuse 设计要点

1. **校准堆叠**：在辅助集上用各组件打分训练逻辑斯蒂元学习器输出 `P(S=1)`。
2. **非劣守卫**：若融合在辅助集上的 AUC 低于最佳组件 `−0.02`，直接回退该组件原始分数——绝不让融合拖后腿。
3. **场景路由**：攻击者无公开特征时，自动降级到 `label_only`，保证任何输入都有可运行路径。
4. **零自研 SOTA**：全部复用 scikit-learn / NumPy 顶级实现；无任何外部权重下载。

## 5. 难度甜点（DGP 旋钮）

`Config` 暴露 `s_mag / s_eff / y_noise / label_noise` 四个难旋钮。`X_pub` 沿方向 `d_x` 按 `S` 平移 `±s_mag`，使 `S` 可从 `X_pub` 部分恢复；`s_eff` 给 `S→Y` 直接效应；`label_noise` 提供过拟合燃料，让被测模型泄露超越标签/特征的敏感信号。旋钮经扫描定在「基线明显低于天花板」区间（见 benchmark.json）。

## 6. 复现与确定性

- 唯一 seed 入口 `core.seed.set_all(seed)`，numpy/random/BLAS 线程一次设齐。
- 同一 seed 两次运行 `benchmark.json` 核心指标**逐位一致**（demo 二次校验）。
- 性能预算：demo 端到端 ≤ 60s（CPU）、内存峰值 ≤ 2GB。
