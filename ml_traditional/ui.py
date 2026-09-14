"""Interface Streamlit — ML tradicional (FLAML + UCI 350)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ml_traditional.data_loader import (
    build_inference_sample,
    dataset_metadata,
    get_column_dictionary,
    load_raw_dataframe,
    prepare_features,
)
from ml_traditional.dataset_catalog import (
    COLUMN_DOCS,
    DATASET_SUMMARY,
    DROP_REASONS,
    FEATURE_PRESETS,
    OPTIONAL_DROP_COLUMNS,
    PAY_STATUS_LEGEND,
    SUGGESTED_DROP_COLUMNS,
    TARGET_COLUMN,
    UCI_PAGE_URL,
)
from ml_traditional.trainer import (
    DEFAULT_META_NAME,
    DEFAULT_MODEL_NAME,
    FlamlConfig,
    load_meta,
    load_model_bundle,
    load_model_from_path,
    predict_new_data,
    save_model_bundle,
    train_flaml_model,
    validate_inference_columns,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml" / "models"

HP_PRESETS: dict[str, dict] = {
    "rapido": {
        "label": "Rápido (60s)",
        "time_budget": 60,
        "estimators": ("rf", "extra_tree", "lrl1"),
        "n_splits": 3,
    },
    "balanceado": {
        "label": "Balanceado (120s) — padrão",
        "time_budget": 120,
        "estimators": ("lgbm", "xgboost", "rf", "extra_tree"),
        "n_splits": 5,
    },
    "profundo": {
        "label": "Profundo (300s)",
        "time_budget": 300,
        "estimators": ("lgbm", "xgboost", "rf", "extra_tree", "lrl1"),
        "n_splits": 5,
    },
}


def _init_ml_state(df: pd.DataFrame) -> None:
    if "ml_raw_df" not in st.session_state:
        st.session_state.ml_raw_df = df
    if "ml_drop_columns" not in st.session_state:
        st.session_state.ml_drop_columns = [
            c for c in SUGGESTED_DROP_COLUMNS if c in df.columns
        ]
    if "ml_trained_bundle" not in st.session_state:
        st.session_state.ml_trained_bundle = None
    if "ml_hp_preset" not in st.session_state:
        st.session_state.ml_hp_preset = "balanceado"


def _render_playbook() -> None:
    with st.expander("Roteiro do cientista de dados (leia antes)", expanded=False):
        st.markdown(
            """
1. **EDA** — confira desbalanceamento (~22% de inadimplentes) e tipos das colunas.
2. **Features** — use preset *Recomendado* (remove `ID`). Ajuste se quiser experimentar.
3. **Treino FLAML** — preset *Balanceado*, métrica `roc_auc`, depois **Treinar**.
4. **Salvar .pkl** — persiste modelo + metadados em `data/ml/models/`.
5. **Predição** — CSV novo **sem** coluna alvo, com as mesmas features do treino.

**Dicionário** de cada coluna está na aba homônima e na tabela de Features.
            """
        )
        st.info(DATASET_SUMMARY)


def _feature_docs_table(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        meta = COLUMN_DOCS.get(col, {})
        if col == TARGET_COLUMN:
            sugestao = "Alvo (não usar como feature)"
        elif col in SUGGESTED_DROP_COLUMNS:
            sugestao = "Remover (recomendado)"
        elif col in OPTIONAL_DROP_COLUMNS:
            sugestao = "Remover (opcional)"
        elif col in drop_columns:
            sugestao = "Será removida"
        else:
            sugestao = "Manter"
        rows.append(
            {
                "coluna": col,
                "tipo": meta.get("tipo", str(df[col].dtype)),
                "descricao": meta.get("descricao", ""),
                "sugestao": sugestao,
                "motivo_remocao": DROP_REASONS.get(col, meta.get("notas", "")),
            }
        )
    return pd.DataFrame(rows)


def _render_column_dictionary() -> None:
    st.markdown("#### Dicionário de colunas (embutido no projeto)")
    st.caption(PAY_STATUS_LEGEND)
    dict_df = get_column_dictionary()
    st.dataframe(dict_df, width="stretch", hide_index=True)
    st.download_button(
        "Baixar dicionário (CSV)",
        data=dict_df.to_csv(index=False).encode("utf-8"),
        file_name="uci350_dicionario_colunas.csv",
        mime="text/csv",
    )
    meta = dataset_metadata()
    st.caption(
        f"Fonte: [{meta['nome']}]({meta['url']}) · "
        f"{meta['linhas']:,} linhas · alvo: `{meta['alvo']}`"
    )


def _render_eda(df: pd.DataFrame) -> None:
    st.markdown("#### Visão geral (EDA)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", f"{len(df):,}")
    c2.metric("Colunas", len(df.columns))
    c3.metric("Inadimplentes (%)", f"{100 * df[TARGET_COLUMN].mean():.2f}%")
    c4.metric("Duplicatas ID", int(df["ID"].duplicated().sum()) if "ID" in df.columns else 0)

    st.caption(DATASET_SUMMARY)
    st.dataframe(df.head(15), width="stretch")

    st.markdown("**Tipos, nulos e cardinalidade**")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "unicos": df.nunique(),
            "descricao": [COLUMN_DOCS.get(c, {}).get("descricao", "") for c in df.columns],
        }
    )
    st.dataframe(summary, width="stretch")

    if TARGET_COLUMN in df.columns:
        st.markdown("**Distribuição do alvo**")
        st.bar_chart(df[TARGET_COLUMN].value_counts().sort_index())


def _apply_feature_preset(preset_key: str, df: pd.DataFrame) -> None:
    preset = FEATURE_PRESETS[preset_key]
    drop_list = [c for c in preset["drop"] if c in df.columns]
    st.session_state.ml_drop_columns = drop_list
    st.toast(f"Preset aplicado: {preset['label']}")


def _render_feature_selection(df: pd.DataFrame) -> list[str]:
    st.markdown("#### Seleção de features")

    st.markdown("**Presets (cientista de dados)**")
    pcols = st.columns(len(FEATURE_PRESETS))
    for col, (key, preset) in zip(pcols, FEATURE_PRESETS.items()):
        with col:
            if st.button(preset["label"], key=f"preset-{key}", width="stretch"):
                _apply_feature_preset(key, df)
            st.caption(str(preset["nota"]))

    st.warning(
        "**Remover sempre:** `ID` (identificador). "
        "**Opcional:** colunas `PAY_*` redundantes se o modelo oscilar no CV."
    )

    feature_candidates = [c for c in df.columns if c != TARGET_COLUMN]
    suggested = st.multiselect(
        "Remover (recomendado)",
        options=[c for c in SUGGESTED_DROP_COLUMNS if c in feature_candidates],
        default=[c for c in st.session_state.ml_drop_columns if c in SUGGESTED_DROP_COLUMNS],
    )
    optional = st.multiselect(
        "Remover (opcional)",
        options=[c for c in OPTIONAL_DROP_COLUMNS if c in feature_candidates],
        default=[c for c in st.session_state.ml_drop_columns if c in OPTIONAL_DROP_COLUMNS],
    )
    custom = st.multiselect(
        "Remover (outras)",
        options=[
            c
            for c in feature_candidates
            if c not in SUGGESTED_DROP_COLUMNS + OPTIONAL_DROP_COLUMNS
        ],
        default=[
            c
            for c in st.session_state.ml_drop_columns
            if c not in SUGGESTED_DROP_COLUMNS + OPTIONAL_DROP_COLUMNS
        ],
    )

    drop_columns = sorted(set(suggested + optional + custom))
    st.session_state.ml_drop_columns = drop_columns

    X_preview, _, feat_cols = prepare_features(df, drop_columns=drop_columns)
    st.success(f"**{len(feat_cols)}** features para treino · alvo: `{TARGET_COLUMN}`")

    st.markdown("**Mapa de colunas (dataset + sugestões)**")
    st.dataframe(_feature_docs_table(df, drop_columns), width="stretch", hide_index=True)

    with st.expander("Prévia da matriz X (5 linhas)", expanded=False):
        st.dataframe(X_preview.head(), width="stretch")

    return drop_columns


def _render_flaml_config() -> tuple[FlamlConfig, float]:
    st.markdown("#### Hiperparâmetros FLAML")

    preset_key = st.radio(
        "Preset de treino",
        options=list(HP_PRESETS.keys()),
        format_func=lambda k: HP_PRESETS[k]["label"],
        horizontal=True,
        key="ml_hp_preset_radio",
    )
    hp = HP_PRESETS[preset_key]

    col_a, col_b, col_c = st.columns(3)
    time_budget = col_a.slider("time_budget (s)", 30, 600, hp["time_budget"], step=30)
    metric = col_b.selectbox(
        "metric",
        ["roc_auc", "f1", "accuracy", "log_loss"],
        index=0,
        help="roc_auc é o padrão para classes desbalanceadas.",
    )
    n_splits = col_c.slider("n_splits (CV)", 2, 10, hp["n_splits"])

    col_d, col_e, col_f, col_g = st.columns(4)
    ensemble = col_d.checkbox("ensemble", value=True)
    early_stop = col_e.checkbox("early_stop", value=True)
    log_training_metric = col_f.checkbox("log_training_metric", value=True)
    eval_method = col_g.selectbox("eval_method", ["cv", "holdout"], index=0)

    estimator_list = st.multiselect(
        "estimator_list",
        options=["lgbm", "xgboost", "rf", "extra_tree", "lrl1", "sgd"],
        default=list(hp["estimators"]),
    )

    with st.expander("Hiperparâmetros dos modelos (custom_hp)", expanded=False):
        col1, col2, col3 = st.columns(3)
        learning_rate = col1.number_input("learning_rate máx (LGBM/XGB)", 0.01, 0.5, 0.1, 0.01)
        n_estimators = col2.number_input("n_estimators máx", 50, 500, 200, 10)
        min_child_samples = col3.number_input("min_child_samples (LGBM)", 5, 100, 20, 5)
        min_sample_split = st.number_input("min_samples_split (RF)", 2, 50, 2, 1)
        max_depth = st.number_input("max_depth XGB (0 = auto)", 0, 20, 0, 1)

    test_size = st.slider("Hold-out para métricas finais (%)", 10, 40, 20, 5) / 100.0
    seed = st.number_input("seed (reprodutibilidade)", 1, 99999, 42, 1)

    return FlamlConfig(
        time_budget=time_budget,
        metric=metric,
        estimator_list=tuple(estimator_list) if estimator_list else ("rf",),
        ensemble=ensemble,
        n_splits=n_splits,
        seed=int(seed),
        eval_method=eval_method,
        early_stop=early_stop,
        log_training_metric=log_training_metric,
        learning_rate=float(learning_rate),
        n_estimators=int(n_estimators),
        min_child_samples=int(min_child_samples),
        min_sample_split=int(min_sample_split),
        max_depth=int(max_depth),
    ), float(test_size)


def _try_load_saved_bundle():
    if st.session_state.ml_trained_bundle is not None:
        return st.session_state.ml_trained_bundle
    if (MODELS_DIR / DEFAULT_MODEL_NAME).is_file():
        try:
            bundle = load_model_bundle()
            st.session_state.ml_trained_bundle = bundle
            return bundle
        except Exception:
            pass
    return None


def _render_training_tab(df: pd.DataFrame, drop_columns: list[str]) -> None:
    config, test_size = _render_flaml_config()

    if st.button("Treinar modelo FLAML", type="primary", width="stretch"):
        with st.spinner("Treinando com FLAML (aguarde o time_budget)..."):
            bundle, X_test, y_test, y_pred, y_proba = train_flaml_model(
                df,
                drop_columns=drop_columns,
                config=config,
                test_size=test_size,
            )
            st.session_state.ml_trained_bundle = bundle

        st.success(f"Melhor estimador: **{bundle.best_estimator}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{bundle.metrics['roc_auc']:.4f}")
        m2.metric("F1", f"{bundle.metrics['f1']:.4f}")
        m3.metric("Accuracy", f"{bundle.metrics['accuracy']:.4f}")
        m4.metric("Linhas teste", bundle.metrics["test_rows"])

        cm = np.array(bundle.metrics["confusion_matrix"])
        st.markdown("**Matriz de confusão (hold-out)**")
        st.dataframe(
            pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]),
            width="stretch",
        )
        with st.expander("Classification report"):
            st.json(bundle.metrics["classification_report"])

    bundle = _try_load_saved_bundle()
    if bundle is not None and st.session_state.get("_last_train_msg") != bundle.trained_at:
        st.info(
            f"Modelo disponível: `{bundle.best_estimator}` · "
            f"treinado em {bundle.trained_at[:19]} · "
            f"ROC-AUC {bundle.metrics['roc_auc']:.4f}"
        )
        st.session_state._last_train_msg = bundle.trained_at

    if bundle is not None:
        st.markdown("#### Exportar modelo (.pkl)")
        if st.button("Salvar bundle em disco", width="stretch"):
            path = save_model_bundle(bundle)
            st.success(f"Salvo: `{path}`")
            st.caption(f"Metadados: `{path.parent / DEFAULT_META_NAME}`")

        model_path = MODELS_DIR / DEFAULT_MODEL_NAME
        if model_path.is_file():
            st.download_button(
                "Baixar .pkl para o computador",
                data=model_path.read_bytes(),
                file_name=DEFAULT_MODEL_NAME,
                mime="application/octet-stream",
                width="stretch",
            )


def _render_predict_tab(df: pd.DataFrame) -> None:
    st.markdown("#### Predição em novos dados")
    st.caption(
        "Dados **fora do dataset de treino**: CSV só com colunas de entrada "
        "(sem `default_payment_next_month`). Colunas faltantes são preenchidas com 0."
    )

    uploaded_pkl = st.file_uploader("Carregar .pkl salvo (opcional)", type=["pkl"])
    if uploaded_pkl is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(uploaded_pkl.getvalue())
            tmp_path = tmp.name
        try:
            bundle = load_model_from_path(tmp_path)
            st.session_state.ml_trained_bundle = bundle
            st.success("Modelo carregado do arquivo enviado.")
        except Exception as exc:
            st.error(f"Pickle inválido: {exc}")

    bundle = _try_load_saved_bundle()
    if bundle is None:
        st.warning("Treine e salve um modelo na aba **Treino FLAML**, ou envie um `.pkl`.")
        return

    meta = load_meta()
    st.markdown("**Features esperadas pelo modelo**")
    st.code(", ".join(meta.get("feature_columns", bundle.feature_columns)))

    template = pd.DataFrame({col: [0] for col in bundle.feature_columns})
    st.download_button(
        "Baixar template CSV (colunas zeradas)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_novos_clientes.csv",
        mime="text/csv",
    )

    demo = build_inference_sample(df, n_rows=5)
    st.download_button(
        "Baixar exemplo (5 clientes do dataset, sem alvo — simula dados novos)",
        data=demo.to_csv(index=False).encode("utf-8"),
        file_name="exemplo_clientes_sem_alvo.csv",
        mime="text/csv",
    )

    st.divider()
    use_demo = st.checkbox("Usar 5 linhas de exemplo do dataset (sem alvo)", value=False)
    uploaded = st.file_uploader("Ou envie seu CSV", type=["csv"])

    infer_df: pd.DataFrame | None = None
    if use_demo:
        infer_df = demo
        st.dataframe(infer_df, width="stretch")
    elif uploaded is not None:
        infer_df = pd.read_csv(uploaded)
        st.dataframe(infer_df.head(10), width="stretch")

    if infer_df is None:
        return

    check = validate_inference_columns(infer_df, bundle)
    if check["missing"]:
        st.warning(
            f"Colunas faltantes ({len(check['missing'])}): serão preenchidas com 0 — "
            + ", ".join(check["missing"][:6])
            + ("..." if len(check["missing"]) > 6 else "")
        )
    if check["extra"]:
        st.info(f"Colunas extras no CSV (ignoradas na predição): {', '.join(check['extra'][:8])}")

    use_threshold = st.checkbox("Aplicar limiar customizado em proba_default", value=False)
    threshold = st.slider("Limiar (proba_default >=)", 0.0, 1.0, 0.5, 0.05) if use_threshold else None

    if st.button("Gerar predições", type="primary", width="stretch"):
        try:
            scored = predict_new_data(
                infer_df,
                bundle,
                proba_threshold=threshold if use_threshold else None,
            )
            st.success(f"Predições para **{len(scored)}** linhas.")
            st.dataframe(scored, width="stretch")
            st.bar_chart(scored["pred_default"].value_counts())
            st.download_button(
                "Baixar CSV com predições",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="predicoes_default_cartao.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Erro na predição: {exc}")


def render_ml_tab() -> None:
    st.subheader("ML tradicional — Default de cartão (UCI 350)")
    st.caption(f"Dataset: [UCI 350]({UCI_PAGE_URL}) · Ferramenta: **FLAML AutoML**")

    try:
        import flaml  # noqa: F401
    except ImportError:
        st.error(
            "FLAML não instalado. Rode **SUBIR.cmd** (Docker) ou "
            "`pip install flaml scikit-learn` no ambiente Python."
        )
        return

    _render_playbook()

    try:
        df = load_raw_dataframe()
    except Exception as exc:
        st.error(f"Falha ao baixar/carregar dataset UCI: {exc}")
        if st.button("Tentar download novamente"):
            load_raw_dataframe(force_download=True)
            st.rerun()
        return

    _init_ml_state(df)

    eda_tab, features_tab, train_tab, predict_tab, docs_tab = st.tabs(
        ["EDA", "Features", "Treino FLAML", "Predição (.pkl)", "Dicionário"]
    )

    with eda_tab:
        _render_eda(df)

    with features_tab:
        _render_feature_selection(df)

    with train_tab:
        _render_training_tab(df, st.session_state.ml_drop_columns)

    with predict_tab:
        _render_predict_tab(df)

    with docs_tab:
        _render_column_dictionary()
