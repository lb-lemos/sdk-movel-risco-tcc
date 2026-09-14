from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Configuração geral do experimento
# ---------------------------------------------------------------------

SEED = 42

N_SESSIONS = 1000
EVENTS_PER_SESSION = 10
N_SUSPICIOUS = 1500

LABEL_COL = "is_suspicious"
GROUP_COL = "session_id"

FEATURE_COLS = [
    "action_type",
    "amount_bucket",
    "hour_of_day",
    "day_of_week",
    "time_since_last_action_s",
    "location_delta_km",
    "network_type",
    "failed_auth_count",
    "device_integrity_flag",
    "sim_swap_flag",
    "new_device_flag",
]

NUMERIC_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "time_since_last_action_s",
    "location_delta_km",
    "failed_auth_count",
    "device_integrity_flag",
    "sim_swap_flag",
    "new_device_flag",
]

CATEGORICAL_FEATURES = [
    "action_type",
    "amount_bucket",
    "network_type",
]

OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------
# Geração determinística do conjunto de dados sintético
# ---------------------------------------------------------------------

def generate_dataset() -> pd.DataFrame:
    """
    Gera 10.000 eventos sintéticos distribuídos em 1.000 sessões.
    A proporção de 15% de eventos suspeitos é exclusivamente experimental
    e não representa prevalência real de fraude ou coação.
    """
    rng = np.random.default_rng(SEED)
    n_events = N_SESSIONS * EVENTS_PER_SESSION

    df = pd.DataFrame(
        {
            GROUP_COL: np.repeat(
                [f"S{i:04d}" for i in range(N_SESSIONS)],
                EVENTS_PER_SESSION,
            )
        }
    )

    df["action_type"] = rng.choice(
        ["LOGIN", "PIX", "TRANSFER", "PASSWORD_CHANGE", "LIMIT_CHANGE"],
        n_events,
        p=[0.28, 0.28, 0.24, 0.10, 0.10],
    )

    amount_bucket: List[str] = []
    for action in df["action_type"]:
        if action in {"PIX", "TRANSFER"}:
            amount_bucket.append(
                str(
                    rng.choice(
                        ["LOW", "MEDIUM", "HIGH"],
                        p=[0.55, 0.32, 0.13],
                    )
                )
            )
        else:
            amount_bucket.append("NONE")

    df["amount_bucket"] = amount_bucket

    df["hour_of_day"] = np.clip(
        np.rint(rng.normal(13.5, 4.0, n_events)),
        0,
        23,
    ).astype(int)

    df["day_of_week"] = rng.integers(
        0,
        7,
        n_events,
    )

    df["time_since_last_action_s"] = np.clip(
        rng.lognormal(
            mean=4.6,
            sigma=0.9,
            size=n_events,
        ),
        1,
        7200,
    ).round(1)

    df["location_delta_km"] = np.clip(
        rng.exponential(
            scale=2.0,
            size=n_events,
        ),
        0,
        100,
    ).round(2)

    df["network_type"] = rng.choice(
        ["WIFI", "CELL"],
        n_events,
        p=[0.62, 0.38],
    )

    df["failed_auth_count"] = np.clip(
        rng.poisson(
            0.25,
            n_events,
        ),
        0,
        5,
    )

    df["device_integrity_flag"] = rng.binomial(
        1,
        0.008,
        n_events,
    )

    df["sim_swap_flag"] = rng.binomial(
        1,
        0.004,
        n_events,
    )

    df["new_device_flag"] = rng.binomial(
        1,
        0.03,
        n_events,
    )

    df[LABEL_COL] = 0

    suspicious_indexes = rng.choice(
        df.index,
        size=N_SUSPICIOUS,
        replace=False,
    )

    df.loc[
        suspicious_indexes,
        LABEL_COL,
    ] = 1

    scenarios = rng.choice(
        [
            "context",
            "auth",
            "integrity",
            "sim",
            "new_device",
            "rapid",
            "composite",
        ],
        size=N_SUSPICIOUS,
        p=[0.22, 0.18, 0.12, 0.10, 0.14, 0.16, 0.08],
    )

    for index, scenario in zip(
        suspicious_indexes,
        scenarios,
    ):
        if scenario == "context":
            df.at[index, "hour_of_day"] = int(
                rng.choice(
                    [0, 1, 2, 3, 4, 5, 22, 23]
                )
            )
            df.at[index, "location_delta_km"] = round(
                float(rng.uniform(25, 250)),
                2,
            )

        elif scenario == "auth":
            df.at[index, "failed_auth_count"] = int(
                rng.integers(3, 7)
            )

        elif scenario == "integrity":
            df.at[index, "device_integrity_flag"] = 1

        elif scenario == "sim":
            df.at[index, "sim_swap_flag"] = 1

        elif scenario == "new_device":
            df.at[index, "new_device_flag"] = 1
            df.at[index, "location_delta_km"] = round(
                float(rng.uniform(10, 120)),
                2,
            )

        elif scenario == "rapid":
            df.at[index, "action_type"] = str(
                rng.choice(
                    ["PIX", "TRANSFER"]
                )
            )
            df.at[index, "amount_bucket"] = str(
                rng.choice(
                    ["LOW", "MEDIUM", "HIGH"],
                    p=[0.45, 0.35, 0.20],
                )
            )
            df.at[index, "time_since_last_action_s"] = round(
                float(rng.uniform(1, 14.9)),
                1,
            )

        elif scenario == "composite":
            df.at[index, "failed_auth_count"] = int(
                rng.integers(3, 7)
            )
            df.at[index, "new_device_flag"] = 1
            df.at[index, "location_delta_km"] = round(
                float(rng.uniform(20, 180)),
                2,
            )

            if rng.random() < 0.5:
                df.at[index, "device_integrity_flag"] = 1
            else:
                df.at[index, "sim_swap_flag"] = 1

    return df


# ---------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------

def build_preprocess() -> ColumnTransformer:
    """
    Cria um pré-processador independente para cada pipeline.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------
# Separação treino / validação / teste por sessão
# ---------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reserva 25% das sessões para teste final.
    Dos 75% restantes, reserva 20% para validação:
    60% treino, 15% validação e 25% teste no total.
    """
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=SEED,
    )

    dev_idx, test_idx = next(
        outer_split.split(
            df,
            groups=df[GROUP_COL],
        )
    )

    development = df.iloc[
        dev_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    train_idx, validation_idx = next(
        inner_split.split(
            development,
            groups=development[GROUP_COL],
        )
    )

    train = development.iloc[
        train_idx
    ].copy()

    validation = development.iloc[
        validation_idx
    ].copy()

    return train, validation, test


# ---------------------------------------------------------------------
# Regras determinísticas
# ---------------------------------------------------------------------

def rules_engine(
    event: pd.Series,
) -> Tuple[float, List[str]]:
    penalty = 0.0
    flags: List[str] = []

    if int(
        event["device_integrity_flag"]
    ) == 1:
        penalty += 0.45
        flags.append(
            "DEVICE_INTEGRITY_RISK"
        )

    if int(
        event["sim_swap_flag"]
    ) == 1:
        penalty += 0.30
        flags.append(
            "SIM_SWAP_RISK"
        )

    if float(
        event["failed_auth_count"]
    ) >= 3:
        penalty += 0.20
        flags.append(
            "AUTH_FRICTION"
        )

    if (
        float(
            event["time_since_last_action_s"]
        ) < 15
        and str(
            event["action_type"]
        ).upper()
        in {"TRANSFER", "PIX"}
    ):
        penalty += 0.15
        flags.append(
            "RAPID_SENSITIVE_SEQUENCE"
        )

    return (
        float(
            min(
                penalty,
                1.0,
            )
        ),
        flags,
    )


# ---------------------------------------------------------------------
# Normalização da pontuação do Isolation Forest
# ---------------------------------------------------------------------

@dataclass
class AnomalyScaler:
    p05: float
    p95: float

    def transform(
        self,
        scores,
    ) -> np.ndarray:
        scores = np.asarray(
            scores,
            dtype=float,
        )

        x = (
            scores - self.p05
        ) / (
            self.p95
            - self.p05
            + 1e-12
        )

        x = np.clip(
            x,
            0.0,
            1.0,
        )

        return 1.0 - x


# ---------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------

def fit_models(
    train_df: pd.DataFrame,
) -> Tuple[
    Pipeline,
    Pipeline,
    AnomalyScaler,
]:
    supervised = Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocess(),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    supervised.fit(
        train_df[FEATURE_COLS],
        train_df[LABEL_COL].astype(int),
    )

    normal_train = train_df[
        train_df[LABEL_COL] == 0
    ].copy()

    anomaly = Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocess(),
            ),
            (
                "iforest",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.05,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    anomaly.fit(
        normal_train[FEATURE_COLS]
    )

    transformed_normal = (
        anomaly
        .named_steps["preprocess"]
        .transform(
            normal_train[
                FEATURE_COLS
            ]
        )
    )

    normal_scores = (
        anomaly
        .named_steps["iforest"]
        .score_samples(
            transformed_normal
        )
    )

    p05, p95 = np.percentile(
        normal_scores,
        [5, 95],
    )

    scaler = AnomalyScaler(
        p05=float(p05),
        p95=float(p95),
    )

    return (
        supervised,
        anomaly,
        scaler,
    )


# ---------------------------------------------------------------------
# Cálculo das pontuações
# ---------------------------------------------------------------------

def compute_scores(
    data: pd.DataFrame,
    supervised: Pipeline,
    anomaly: Pipeline,
    scaler: AnomalyScaler,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    supervised_score = (
        supervised.predict_proba(
            data[FEATURE_COLS]
        )[:, 1]
    )

    transformed = (
        anomaly
        .named_steps["preprocess"]
        .transform(
            data[FEATURE_COLS]
        )
    )

    raw_anomaly = (
        anomaly
        .named_steps["iforest"]
        .score_samples(
            transformed
        )
    )

    anomaly_score = scaler.transform(
        raw_anomaly
    )

    rule_score = np.array(
        [
            rules_engine(row)[0]
            for _, row
            in data.iterrows()
        ],
        dtype=float,
    )

    hybrid_score = (
        0.45 * supervised_score
        + 0.35 * anomaly_score
        + 0.20 * rule_score
    )

    return (
        supervised_score,
        anomaly_score,
        rule_score,
        hybrid_score,
    )


# ---------------------------------------------------------------------
# Calibração do limiar apenas no conjunto de validação
# ---------------------------------------------------------------------

def choose_threshold_by_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2
        * precision[:-1]
        * recall[:-1]
    ) / (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )

    best_index = int(
        np.nanargmax(
            f1_values
        )
    )

    return float(
        thresholds[
            best_index
        ]
    )


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
    prediction = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


# ---------------------------------------------------------------------
# Geração das flags explicativas
# ---------------------------------------------------------------------

def event_flags(
    event: pd.Series,
    anomaly_score: float,
    anomaly_threshold: float = 0.60,
) -> List[str]:
    _, flags = rules_engine(
        event
    )

    if anomaly_score >= (
        anomaly_threshold
    ):
        flags.append(
            "CONTEXT_ANOMALY"
        )

    return sorted(
        set(flags)
    )


# ---------------------------------------------------------------------
# Inferência de um único evento
# ---------------------------------------------------------------------

def compute_single_event(
    event: pd.Series,
    supervised: Pipeline,
    anomaly: Pipeline,
    scaler: AnomalyScaler,
) -> Tuple[
    int,
    List[str],
    Dict[str, float],
]:
    event_df = pd.DataFrame(
        [event[FEATURE_COLS]]
    )

    supervised_score = float(
        supervised.predict_proba(
            event_df
        )[0, 1]
    )

    transformed = (
        anomaly
        .named_steps["preprocess"]
        .transform(
            event_df
        )
    )

    raw_anomaly = float(
        anomaly
        .named_steps["iforest"]
        .score_samples(
            transformed
        )[0]
    )

    anomaly_score = float(
        scaler.transform(
            [raw_anomaly]
        )[0]
    )

    rule_score, flags = rules_engine(
        event
    )

    if anomaly_score >= 0.60:
        flags.append(
            "CONTEXT_ANOMALY"
        )

    combined = (
        0.45 * supervised_score
        + 0.35 * anomaly_score
        + 0.20 * rule_score
    )

    risk_score = int(
        np.clip(
            np.rint(
                combined * 100
            ),
            0,
            100,
        )
    )

    breakdown = {
        "supervised_score":
            supervised_score,
        "anomaly_score":
            anomaly_score,
        "rule_score":
            rule_score,
        "hybrid_score":
            float(combined),
    }

    return (
        risk_score,
        sorted(
            set(flags)
        ),
        breakdown,
    )


# ---------------------------------------------------------------------
# Medição de latência no protótipo de laboratório
# ---------------------------------------------------------------------

def measure_latency(
    data: pd.DataFrame,
    supervised: Pipeline,
    anomaly: Pipeline,
    scaler: AnomalyScaler,
    max_events: int = 200,
) -> float:
    sample = data.head(
        min(
            len(data),
            max_events,
        )
    )

    start = time.perf_counter()

    for _, row in sample.iterrows():
        compute_single_event(
            row,
            supervised,
            anomaly,
            scaler,
        )

    end = time.perf_counter()

    return (
        (end - start)
        / len(sample)
        * 1000
    )


# ---------------------------------------------------------------------
# Persistência dos resultados
# ---------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    supervised: Pipeline,
    anomaly: Pipeline,
    scaler: AnomalyScaler,
    thresholds: Dict[str, float],
    results: Dict[
        str,
        Dict[str, float | int],
    ],
    test_scores: Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    latency_ms: float,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        OUTPUT_DIR
        / "sdk_events_dataset.csv"
    )

    df.to_csv(
        dataset_path,
        index=False,
    )

    (
        supervised_score,
        anomaly_score,
        rule_score,
        hybrid_score,
    ) = test_scores

    output = test.copy()

    output[
        "supervised_score"
    ] = supervised_score

    output[
        "anomaly_score"
    ] = anomaly_score

    output[
        "rule_score"
    ] = rule_score

    output[
        "hybrid_score"
    ] = hybrid_score

    output[
        "risk_score"
    ] = np.clip(
        np.rint(
            hybrid_score * 100
        ),
        0,
        100,
    ).astype(int)

    output[
        "flags"
    ] = [
        ";".join(
            event_flags(
                row,
                anomaly_value,
            )
        )
        for (
            (_, row),
            anomaly_value,
        )
        in zip(
            test.iterrows(),
            anomaly_score,
        )
    ]

    output.to_csv(
        OUTPUT_DIR
        / "sdk_test_results.csv",
        index=False,
    )

    metrics_rows = []

    for model_name, model_metrics in (
        results.items()
    ):
        row = {
            "model":
                model_name,
            **model_metrics,
        }
        metrics_rows.append(
            row
        )

    pd.DataFrame(
        metrics_rows
    ).to_csv(
        OUTPUT_DIR
        / "metrics.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "thresholds.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            thresholds,
            file,
            ensure_ascii=False,
            indent=2,
        )

    split_summary = {
        "seed": SEED,
        "total_events": int(
            len(df)
        ),
        "total_sessions": int(
            df[GROUP_COL].nunique()
        ),
        "training_events": int(
            len(train)
        ),
        "validation_events": int(
            len(validation)
        ),
        "test_events": int(
            len(test)
        ),
        "test_reference_events": int(
            (
                test[LABEL_COL] == 0
            ).sum()
        ),
        "test_suspicious_events": int(
            (
                test[LABEL_COL] == 1
            ).sum()
        ),
        "average_latency_ms_per_event":
            float(latency_ms),
    }

    with open(
        OUTPUT_DIR
        / "split_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------
# Execução completa
# ---------------------------------------------------------------------

def main() -> None:
    # 1. Geração do conjunto experimental
    df = generate_dataset()

    # 2. Separação por sessão
    train, validation, test = (
        split_dataset(df)
    )

    # 3. Treinamento
    supervised, anomaly, scaler = (
        fit_models(train)
    )

    # 4. Scores da validação
    (
        val_supervised,
        val_anomaly,
        _,
        val_hybrid,
    ) = compute_scores(
        validation,
        supervised,
        anomaly,
        scaler,
    )

    y_validation = (
        validation[
            LABEL_COL
        ]
        .astype(int)
        .to_numpy()
    )

    # 5. Calibração dos limiares apenas na validação
    thresholds = {
        "Random Forest":
            choose_threshold_by_f1(
                y_validation,
                val_supervised,
            ),
        "Isolation Forest":
            choose_threshold_by_f1(
                y_validation,
                val_anomaly,
            ),
        "Motor híbrido":
            choose_threshold_by_f1(
                y_validation,
                val_hybrid,
            ),
    }

    # 6. Scores do teste final
    test_scores = compute_scores(
        test,
        supervised,
        anomaly,
        scaler,
    )

    (
        test_supervised,
        test_anomaly,
        _,
        test_hybrid,
    ) = test_scores

    y_test = (
        test[
            LABEL_COL
        ]
        .astype(int)
        .to_numpy()
    )

    # 7. Métricas finais
    results = {
        "Random Forest":
            calculate_metrics(
                y_test,
                test_supervised,
                thresholds[
                    "Random Forest"
                ],
            ),
        "Isolation Forest":
            calculate_metrics(
                y_test,
                test_anomaly,
                thresholds[
                    "Isolation Forest"
                ],
            ),
        "Motor híbrido":
            calculate_metrics(
                y_test,
                test_hybrid,
                thresholds[
                    "Motor híbrido"
                ],
            ),
    }

    # 8. Latência do protótipo
    latency_ms = measure_latency(
        test,
        supervised,
        anomaly,
        scaler,
    )

    # 9. Persistência dos artefatos
    save_outputs(
        df=df,
        train=train,
        validation=validation,
        test=test,
        supervised=supervised,
        anomaly=anomaly,
        scaler=scaler,
        thresholds=thresholds,
        results=results,
        test_scores=test_scores,
        latency_ms=latency_ms,
    )

    # 10. Saída resumida
    print(
        "Treinamento:",
        len(train),
    )
    print(
        "Validação:",
        len(validation),
    )
    print(
        "Teste:",
        len(test),
    )
    print(
        "Eventos de referência no teste:",
        int(
            (
                test[LABEL_COL] == 0
            ).sum()
        ),
    )
    print(
        "Eventos suspeitos no teste:",
        int(
            (
                test[LABEL_COL] == 1
            ).sum()
        ),
    )

    print(
        "\nLimiares calibrados na validação:"
    )
    for name, threshold in (
        thresholds.items()
    ):
        print(
            f"{name}: "
            f"{threshold:.6f}"
        )

    print(
        "\nMétricas no teste final:"
    )
    for name, model_metrics in (
        results.items()
    ):
        print(
            f"\n{name}"
        )
        for key, value in (
            model_metrics.items()
        ):
            if isinstance(
                value,
                float,
            ):
                print(
                    f"  {key}: "
                    f"{value:.6f}"
                )
            else:
                print(
                    f"  {key}: "
                    f"{value}"
                )

    print(
        "\nLatência média do protótipo:"
    )
    print(
        f"{latency_ms:.3f} ms/evento"
    )

    print(
        "\nArquivos gerados em:"
    )
    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()
