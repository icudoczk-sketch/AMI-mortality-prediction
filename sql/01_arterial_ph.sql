-- 01_arterial_ph.sql
-- 目的：重新提取「动脉血气 pH」，替换当前混入非血气标本的 pH 列。
-- 背景：现稿 pH 呈双峰（中位 6.50，最小 5.00，51% 记录 <6.5），
--       提示按标签("pH")匹配时把尿/胸水等标本一并纳入。血气 pH 不可能 <6.8。
-- 适用：MIMIC-IV v2.2（PostgreSQL）。
-- 依赖：mimiciv_icu.icustays、mimiciv_derived.bg（若已构建 derived concepts）。
--       若未构建，见文件末尾的等效原始表写法。

-- ============================================================
-- 步骤 0：先证明问题存在（对照用，不改数据）
-- ============================================================
-- 按标本类型看 pH 分布：ART 应集中在 7.35–7.45；若 ART 也出现 5.x，说明标本标注不可靠，
-- 需改用 itemid 50820 且仅保留动脉血气标本（见步骤 2 的备选写法）。
SELECT specimen,
       count(*)                                              AS n,
       percentile_cont(0.25) WITHIN GROUP (ORDER BY ph)      AS q1,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY ph)      AS median,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY ph)      AS q3,
       min(ph)                                               AS min_ph,
       max(ph)                                               AS max_ph
FROM mimiciv_derived.bg
WHERE ph IS NOT NULL
GROUP BY specimen
ORDER BY n DESC;

-- ============================================================
-- 步骤 1：按 stay 取「入 ICU 后首个 24 小时内、最早一次」动脉血气 pH
-- ============================================================
WITH ph_art AS (
    SELECT c.stay_id,
           b.charttime,
           b.ph,
           ROW_NUMBER() OVER (PARTITION BY c.stay_id ORDER BY b.charttime) AS rn
    FROM mimiciv_icu.icustays c
    JOIN mimiciv_derived.bg b
      ON  b.subject_id = c.subject_id
      AND b.charttime  >= c.intime
      AND b.charttime  <  c.intime + INTERVAL '24 hour'
    WHERE b.specimen = 'ART'
      AND b.ph IS NOT NULL
      AND b.ph BETWEEN 6.80 AND 7.80          -- 生理范围硬约束，剔除明显错误录入
)
SELECT stay_id,
       ph        AS ph_arterial_first24h,
       charttime AS ph_charttime
FROM ph_art
WHERE rn = 1
ORDER BY stay_id;

-- ============================================================
-- 步骤 2：备选写法（未构建 derived 概念时，直接读 labevents）
-- 注意：labevents 无 specimen 字段，必须借 d_labitems 的 label 与 fluid 过滤；
--       若 fluid 仍无法区分，改用 itemid 50820 并配合 chartevents 的血气记录交叉核对。
-- ============================================================
-- SELECT le.subject_id, le.hadm_id, le.charttime, le.valuenum AS ph
-- FROM mimiciv_hosp.labevents le
-- JOIN mimiciv_hosp.d_labitems d ON d.itemid = le.itemid
-- WHERE d.label = 'pH'
--   AND d.fluid = 'Blood'          -- 关键：排除 Urine / Pleural / Ascites 等
--   AND le.valuenum BETWEEN 6.80 AND 7.80;

-- ============================================================
-- 步骤 3：导出后与现有分析表按 stay_id 合并，替换 pH 列，再重跑建模。
-- 校验：新 pH 中位数应落在 7.30–7.42 之间，Q1 ≥ 7.20，最小值 ≥ 6.80。
-- ============================================================
