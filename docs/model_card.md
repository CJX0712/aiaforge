# AiaForge · 模型卡（Model Card）

> 系统：AiaForge — 属性推断攻击基准（旗舰 AiaFuse）
> 作者：晨星 (CJX0712)

## 模型细节

- **类型**：对抗性隐私攻击工具箱（属性推断），非预测模型本身。
- **旗舰**：`AiaFuse` — 校准堆叠融合 + 非劣守卫 + 场景路由。
- **依赖**：scikit-learn 1.9.1 / NumPy 2.5.4 / SciPy 1.18.1（离线兜底纯 NumPy）。
- **训练数据**：合成表格数据（固定 seed 可复现），无真实个人数据。

## 预期用途

- 红队/隐私合规诊断：量化「部署模型泄露个体敏感属性」的风险。
- 研究：对比不同目标模型（MLP / LogReg）在过拟合下的属性泄漏程度。
- **禁止用途**：未经授权推断真实个体敏感属性（违反隐私法）。

## 评估

- 指标：ROC-AUC（主）、Accuracy（决策阈 0.5）。
- 协议：train / aux / test 三分区严格隔离，≥3 seeds 报 mean±std，胜基线判据 `diff > ½(σ₁+σ₂)`。
- 结果：见 `benchmark.json` 与 README 性能基线表。

## 局限与诚实声明

- 合成为可控 DGP，绝对数值不代表真实数据集上的风险量级。
- `attack_nn` 在无 scikit-learn 时退化为 NumPy 逻辑回归（已加单测）。
- 若被测模型未过拟合（无 `label_noise` 或 `S` 与 `Y` 弱相关），泄漏信号减弱属预期，非实现缺陷。
- 失败案例分析（典型误例）从 `benchmark.json` 的 `failures` 派生，非编造。
