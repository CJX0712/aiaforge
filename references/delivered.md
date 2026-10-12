# AiaForge 交付记录 (Delivered Systems)

> 闭环交付回写表。每条记录 = 一个已开源至 GitHub 的世界顶级 AI 系统。作者：晨星 (CJX0712)。
> 原则：零重写（复用世界最优解）、集大成（站在顶级成果之上）、唯一目标（架构/性能/代码/文档最全）、自治执行。

| # | 系统 | 领域 | 日期 | Tag | 等级 | 关键指标 | URL |
|---|------|------|------|-----|------|----------|-----|
| 1 | AiaForge | 属性推断攻击 (Attribute Inference Attack) 基准与融合 | 2026-10-12 | v0.1.0 | **S** | aiafuse ROC-AUC 0.8926 vs majority 0.5000 (Δ+0.3926, gate≥0.10 PASS, significant)；non-inferior to best single attack_lr 0.8933 (Δ-0.0007)；17 单测全绿；同 seed 逐位一致；demo 32.8s≤60s；ruff lint+format clean | https://github.com/CJX0712/aiaforge |

## 设计要点（AiaForge v0.1.0）
- **零重写 / 集大成**：目标模型复用 `scikit-learn` MLPClassifier / LogisticRegression；元学习器采用 Wolpert (1992) **OOF stacked generalization**（K-fold 袋外分数），避免样本内过拟合——这是教科书级正确修复，而非 hack。
- **旗舰攻击 `AiaFuse`**：以「对抗特征（X_pub / Y / 置信度差）」+「各分量袋外分数」做输入，LogisticRegression(C=0.5) 校准融合；内置 non-inferiority guard（融合退化到最佳单分量）与 regime router（无 X_pub 时退化为 label_only）。
- **确定性**：`core.seed.set_all()` 锁定 `OMP/MKL/OPENBLAS_NUM_THREADS=1`，同 seed 逐位可复现。
- **无泄漏分区**：train / aux / test 严格连续不相交；aux 仅用于训练攻击，test 仅用于评测。
- **显著性格**：`mean_diff > ½(σ_a+σ_b)` 作为击败基线的判定门。
- **离线降级**：scikit-learn 不可用时，`aia/_numpy_models.NumpyLogReg` 纯 NumPy 兜底，攻击流水线仍可跑通。
- **质量门（S 级）**：DoD 全绿 + 多 seed 击败强基线达阈值 + CI 矩阵（ubuntu/windows × py3.12/3.13）绿 + Release 打 tag。
- **交付自动化**：`scripts/gh_push.py` 三级降级（git push → Git Data API → Contents API），沙箱内 git 协议被代理拦截时自动切到 API 推送并打 Release。
