# Changelog

## v0.1.0 (2026-10-12)

- 初始发布：AiaForge — 世界顶级属性推断攻击（Attribute Inference Attack）系统。
- 旗舰 `AiaFuse`：校准堆叠融合 + 非劣守卫 + 场景路由（无 `X_pub` 时退 `label_only`）。
- 攻击方法：majority / label_only / conf_inversion(Yeom 式置信反演) / attack_lr(元分类器) / attack_nn(MLP 元分类器) / aiafuse(旗舰)。
- 被测目标模型：scikit-learn MLP / LogisticRegression，纯 NumPy 离线兜底。
- 合成 DGP 完全可复现；train/aux/test 三分区严格隔离，无泄漏。
- ≥3 seeds 报 mean±std；胜基线判据 `diff > ½(σ₁+σ₂)`；同 seed 两次运行核心指标逐位一致。
- 失败案例分析从 `benchmark.json` 派生，不编造。
- 质量等级：见 README 性能基线表（以 benchmark.json 实测为准）。
- 作者：晨星 (CJX0712)。
