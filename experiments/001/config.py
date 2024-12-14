import dotenv

dotenv.load_dotenv()

ENVIRONMENT = "local"
EXPERIMENT = "001"

if ENVIRONMENT == "local":
    import rootutils

    ROOT_DIR = rootutils.setup_root(".", cwd=True, pythonpath=True, dotenv=True)
    DATA_PATH = ROOT_DIR / "data"
    INPUT_DIR = DATA_PATH / "inputs"
    OUTPUT_DIR = DATA_PATH / "outputs" / EXPERIMENT

COMPETITION_DATASET_DIR = INPUT_DIR / "equity-post-HCT-survival-predictions"

for d in [INPUT_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    "cox_target",
]
