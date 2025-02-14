import os
import sys
from pathlib import Path

EXP_NAME = "129"

N_SPLITS = 10
SEED = 1
SEEDS = [SEED + i for i in range(5)]

VALID_RATIO: float | None = None


FEATURE_PREFIX = "f_"

DATASET_COL = "dataset"
FOLD_COL = "fold"
ID_COL = "ID"

CATEGORICAL_COLS = [
    "dri_score",
    "psych_disturb",
    "cyto_score",
    "diabetes",
    "tbi_status",
    "arrhythmia",
    "graft_type",
    "vent_hist",
    "renal_issue",
    "pulm_severe",
    "prim_disease_hct",
    "cmv_status",
    "tce_imm_match",
    "rituximab",
    "prod_type",
    "cyto_score_detail",
    "conditioning_intensity",
    "ethnicity",
    "obesity",
    "mrd_hct",
    "in_vivo_tcd",
    "tce_match",
    "hepatic_severe",
    "prior_tumor",
    "peptic_ulcer",
    "gvhd_proph",
    "rheum_issue",
    "sex_match",
    "race_group",
    "hepatic_mild",
    "tce_div_match",
    "donor_related",
    "melphalan_dose",
    "cardiac",
    "pulm_moderate",
]

NUMERICAL_COLS = [
    "hla_match_c_high",
    "hla_high_res_8",
    "hla_low_res_6",
    "hla_high_res_6",
    "hla_high_res_10",
    "hla_match_dqb1_high",
    "hla_nmdp_6",
    "hla_match_c_low",
    "hla_match_drb1_low",
    "hla_match_dqb1_low",
    "year_hct",
    "hla_match_a_high",
    "donor_age",
    "hla_match_b_low",
    "age_at_hct",
    "hla_match_a_low",
    "hla_match_b_high",
    "comorbidity_score",
    "karnofsky_score",
    "hla_low_res_8",
    "hla_match_drb1_high",
    "hla_low_res_10",
]

EVENT_COL = "efs"
SURVIVAL_TIME_COL = "efs_time"
META_COLS = [
    ID_COL,
    DATASET_COL,
    FOLD_COL,
    EVENT_COL,
    SURVIVAL_TIME_COL,
    "race_group",
]


# ---------- # DIRECTORIES # ---------- #
IS_KAGGLE_ENV = os.getenv("KAGGLE_DATA_PROXY_TOKEN") is not None
KAGGLE_COMPETITION_NAME = os.getenv("KAGGLE_COMPETITION_NAME", "equity-post-HCT-survival-predictions")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "mst8823")
EXP_DIR = Path(sys.modules[__name__].__file__).resolve().parent

if not IS_KAGGLE_ENV:
    import rootutils

    sys.path.append(str(EXP_DIR))  # add pythonpath for the current experiment

    ROOT_DIR = rootutils.setup_root(
        ".",
        indicator="pyproject.toml",
        cwd=True,
        pythonpath=True,
    )
    INPUT_DIR = ROOT_DIR / "data" / "input"
    # RTIFACT_DIR / EXP_NAME / 1 でのアクセスを想定
    ARTIFACT_DIR = ROOT_DIR / "data" / "output"
    # 当該 code の生成物の出力先. kaggle code とパスを合わせるために 1 を付与
    OUTPUT_DIR = ARTIFACT_DIR / EXP_NAME / "1"
    EXP_TMP_DIR = ROOT_DIR / "data" / "tmp" / EXP_NAME / "1"  # 実験ごとの一時ファイルを格納する場所
    TMP_DIR = ROOT_DIR / "data" / "tmp" / "common"  # 共有の一時ファイルを格納する場所

    ARTIFACTS_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-artifacts/other/{EXP_NAME}"
    CODES_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-codes"
else:
    sys.path.append(str(EXP_DIR))  # add pythonpath for the current experiment

    ROOT_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")
    # 当該 code 以外の生成物が格納されている場所 (Model として使用できる)  ARTIFACT_DIR / EXP_NAME / 1 でアクセス可能
    ARTIFACT_DIR = INPUT_DIR / f"{KAGGLE_COMPETITION_NAME.lower()}-artifacts" / "other"
    OUTPUT_DIR = TMP_DIR = EXP_TMP_DIR = ROOT_DIR  # 当該 code の生成物の出力先


COMP_DATASET_DIR = INPUT_DIR / KAGGLE_COMPETITION_NAME

for d in [INPUT_DIR, OUTPUT_DIR, EXP_TMP_DIR]:
    d.mkdir(exist_ok=True, parents=True)

ARTIFACT_EXP_DIR = lambda exp_name=EXP_NAME: ARTIFACT_DIR / exp_name / "1"  # noqa  # 対象の exp の artifact が格納されている場所を返す
