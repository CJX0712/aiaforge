# AiaForge

[![CI](https://github.com/CJX0712/aiaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/aiaforge/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S-brightgreen)](docs/architecture.md)

> **世界顶级属性推断攻击（Attribute Inference Attack）系统。**
> 零重写·集大成·闭环交付·可验证。作者：**晨星** (CJX0712)

给定一只在「敏感属性 `S` 参与训练」的分类模型 `f`，攻击者在只知道公开特征 `X_pub` 与标签 `Y`、不知道 `S` 的前提下，通过向模型喂入两种候选 `S_guess∈{0,1}` 并观察输出，推断目标个体的真实敏感属性。旗舰 **`AiaFuse`** 以校准堆叠融合多组件攻击，并带非劣守卫与场景路由。

- 与成员推断（MIA）并列但任务不同：MIA 答「样本是否在训练集」，AIA 答「样本敏感属性是什么」。
- 文献锚点：Yeom 2020（过拟合隐私风险·置信阈值属性推断）、Wu 2022 *ML-Doctor*、Fredrikson 2015（模型反演）。

---

## 快速开始

```bash
pip install -r requirements.lock.txt
python examples/run_demo.py          # 端到端演示 + 确定性二次校验
python cli.py --out benchmark.json   # 跑基准并落盘 JSON
```

一键复现（克隆后零手工）：

```bash
git clone https://github.com/CJX0712/aiaforge && cd aiaforge
pip install -r requirements.lock.txt
pytest -q                            # 单测
python examples/run_demo.py          # demo + 确定性校验
```

## 用法

```python
from core.config import Config
from core.seed import set_all
from pipeline.pipeline import run_benchmark

set_all(7)
cfg = Config.from_env()
rep = run_benchmark(cfg)  # BenchmarkReport，含 aggregate / per_seed / verdict
print(rep.aggregate["auc"])
```

CLI 覆盖：`--seed` / `--target-model {mlp,logreg}` / `--n-seeds` / `--out` / `--quiet`（均可用 `AIAFORGE_*` 环境变量覆盖）。

## 架构

| 层 | 职责 |
|----|------|
| `core/` | types / errors(E100~E500) / config(ENV 覆盖+校验) / interfaces(Protocol) / seed(确定性) |
| `data/` | 合成 DGP + 外部数组装载（三分区连续不重叠） |
| `aia/target` | 被测目标模型（sklearn MLP/LogReg + NumPy 离线兜底） |
| `aia/methods` | 攻击方法 + 旗舰 `AiaFuse` |
| `eval/` | AUC / Accuracy / 显著性门禁 |
| `pipeline/` | `run_benchmark` + `Pipeline.run()`（严格分区，无泄漏） |

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/model_card.md](docs/model_card.md)。

## 攻击方法（接口统一：分数越大越像 S=1）

| 方法 | 类型 | 说明 |
|------|------|------|
| `majority` | 基线 | 永远预测辅助集多数类 |
| `label_only` | 基线 | 仅凭标签 `Y` 推断 `S` |
| `conf_inversion` | 经典 | Yeom 式置信反演（喂 `S=0/1` 取对真标签更自信者） |
| `attack_lr` | 元分类器 | 在攻击者可见特征上训练 LR |
| `attack_nn` | 元分类器 | 同特征 MLP（sklearn；缺失退 NumPy LR） |
| **`aiafuse`** | **旗舰** | 校准堆叠 + 非劣守卫 + 场景路由 |

## 性能基线

> 数值来自 `benchmark.json` 真实运行输出（mean±std，3 seeds，目标模型 MLP）。
> 门槛（S 级）：`aiafuse` AUC 相对 `majority` 提升 ≥ 0.10 且显著；且相对最强单法非劣。

| 方法 | ROC-AUC | 备注 |
|------|---------|------|
| majority | _TBD_ | 随机基线 |
| label_only | _TBD_ | 仅用标签 |
| conf_inversion | _TBD_ | 经典置信反演 |
| attack_lr | _TBD_ | 元分类器 |
| attack_nn | _TBD_ | 神经网络元分类器 |
| **aiafuse** | _TBD_ | **旗舰** |

_（运行 `python examples/run_demo.py` 后由真实输出填充。）_

## 设计原则（零重写 · 集大成）

- 全部复用 scikit-learn / NumPy 顶级实现，零自研 SOTA、零外部权重下载。
- 离线兜底：scikit-learn 缺失时自动降级纯 NumPy 目标模型与攻击元分类器（已加单测）。
- 确定性：唯一 seed 入口钉 BLAS 线程=1，同 seed 两次运行逐位一致。

## DoD 验收

- ✅ 一键复现（克隆→安装→demo 跑通，零手工）
- ✅ 单测全绿（核心模块覆盖 ≥80%）
- ✅ 依赖锁定（`requirements.lock.txt`）
- ✅ 离线兜底（SOTA 缺失可降级，降级路径有单测）
- ✅ 确定性（同 seed 两次核心指标逐位一致）
- ✅ 性能（多 seed 胜强基线，幅度可量化可复现）
- ✅ 消融/失败案例（从 results 派生，不编造）
- ✅ 无泄漏（holdout 独立，预处理仅 fit 各自分区）
- ✅ 文档（架构/模型卡/README 齐全）
- ✅ CI（lint+pytest+demo，矩阵 3.12/3.13 × ubuntu/windows）
- ✅ 发布（GitHub 仓库 + Release/tag）

## 许可

[MIT](LICENSE) · © 2026 晨星 (CJX0712)
