"""Metadados do UCI 350 — Default of Credit Card Clients (Taiwan)."""

from __future__ import annotations

TARGET_COLUMN = "default_payment_next_month"
TARGET_COLUMN_RAW = "default payment next month"

UCI_PAGE_URL = (
    "https://archive-beta.ics.uci.edu/dataset/350/default+of+credit+card+clients"
)
# Download oficial (UCI beta / CDN)
UCI_BETA_ZIP_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/350/default+of+credit+card+clients.zip"
UCI_XLS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# Nomes anonimos (X1..X23, Y) no portal beta → nomes usados no app
BETA_ANON_RENAME: dict[str, str] = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET_COLUMN,
}

DATASET_SUMMARY = (
    "30.000 clientes de cartão de crédito (Taiwan, 2005). "
    "Objetivo: prever inadimplência no mês seguinte (classe minoritária ~22%). "
    "Recomendação: métrica **roc_auc** ou **f1**, remover **ID**, validar com hold-out + CV."
)

# Colunas que DEVEM ser removidas antes do treino
SUGGESTED_DROP_COLUMNS = ["ID"]

# Colunas opcionais — redundância / experimentação
OPTIONAL_DROP_COLUMNS = [
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]

# Motivo de remoção (exibido na UI)
DROP_REASONS: dict[str, str] = {
    "ID": "Identificador — sem poder preditivo; causa overfitting aparente.",
    "PAY_0": "Histórico de pagamento — correlacionado com PAY_2..6; teste remover para simplificar.",
    "PAY_2": "Idem PAY_0 — série temporal redundante.",
    "PAY_3": "Idem PAY_0.",
    "PAY_4": "Idem PAY_0.",
    "PAY_5": "Idem PAY_0.",
    "PAY_6": "Idem PAY_0.",
}

# Presets de features (cientista de dados)
FEATURE_PRESETS: dict[str, dict[str, str | list[str]]] = {
    "recomendado": {
        "label": "Recomendado (só remove ID)",
        "drop": ["ID"],
        "nota": "Ponto de partida padrão para modelagem.",
    },
    "completo": {
        "label": "Todas as features (exceto alvo)",
        "drop": [],
        "nota": "Inclui ID — útil só para diagnóstico, não para produção.",
    },
    "conservador": {
        "label": "Conservador (ID + PAY_* antigos)",
        "drop": ["ID", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
        "nota": "Mantém PAY_0 (status mais recente) e faturas/pagamentos.",
    },
}

COLUMN_DOCS: dict[str, dict[str, str]] = {
    "ID": {
        "descricao": "Identificador do cliente no dataset original.",
        "tipo": "int",
        "notas": "Sem valor preditivo. Remover antes do treino.",
    },
    "LIMIT_BAL": {
        "descricao": "Limite de crédito concedido (NT$, Taiwan).",
        "tipo": "numeric",
        "notas": "Escala ampla; modelos baseados em árvore lidam bem sem padronizar.",
    },
    "SEX": {
        "descricao": "Sexo: 1 = masculino, 2 = feminino.",
        "tipo": "categorical",
        "notas": "",
    },
    "EDUCATION": {
        "descricao": "Escolaridade: 1 = pós-graduação, 2 = universidade, 3 = ensino médio, 4 = outros.",
        "tipo": "categorical",
        "notas": "Valores 0, 5 e 6 aparecem no dado bruto (não documentados oficialmente).",
    },
    "MARRIAGE": {
        "descricao": "Estado civil: 1 = casado, 2 = solteiro, 3 = outros.",
        "tipo": "categorical",
        "notas": "Valor 0 aparece no dado bruto (não documentado).",
    },
    "AGE": {
        "descricao": "Idade do titular em anos.",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_0": {
        "descricao": "Status de pagamento em set/2005.",
        "tipo": "ordinal",
        "notas": "-2=sem consumo, -1=pago em dia, 0=adiantamento, 1..9=atraso (meses).",
    },
    "PAY_2": {
        "descricao": "Status de pagamento em ago/2005.",
        "tipo": "ordinal",
        "notas": "Mesma codificação de PAY_0.",
    },
    "PAY_3": {
        "descricao": "Status de pagamento em jul/2005.",
        "tipo": "ordinal",
        "notas": "Mesma codificação de PAY_0.",
    },
    "PAY_4": {
        "descricao": "Status de pagamento em jun/2005.",
        "tipo": "ordinal",
        "notas": "Mesma codificação de PAY_0.",
    },
    "PAY_5": {
        "descricao": "Status de pagamento em mai/2005.",
        "tipo": "ordinal",
        "notas": "Mesma codificação de PAY_0.",
    },
    "PAY_6": {
        "descricao": "Status de pagamento em abr/2005.",
        "tipo": "ordinal",
        "notas": "Mesma codificação de PAY_0.",
    },
    "BILL_AMT1": {
        "descricao": "Valor da fatura em set/2005 (NT$).",
        "tipo": "numeric",
        "notas": "Pode conter negativos (crédito/ajustes).",
    },
    "BILL_AMT2": {
        "descricao": "Valor da fatura em ago/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "BILL_AMT3": {
        "descricao": "Valor da fatura em jul/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "BILL_AMT4": {
        "descricao": "Valor da fatura em jun/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "BILL_AMT5": {
        "descricao": "Valor da fatura em mai/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "BILL_AMT6": {
        "descricao": "Valor da fatura em abr/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_AMT1": {
        "descricao": "Valor pago em set/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_AMT2": {
        "descricao": "Valor pago em ago/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_AMT3": {
        "descricao": "Valor pago em jul/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_AMT4": {
        "descricao": "Valor pago em jun/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_AMT5": {
        "descricao": "Valor pago em mai/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    "PAY_AMT6": {
        "descricao": "Valor pago em abr/2005 (NT$).",
        "tipo": "numeric",
        "notas": "",
    },
    TARGET_COLUMN: {
        "descricao": "Inadimplência no mês seguinte: 1 = sim (default), 0 = não.",
        "tipo": "binary",
        "notas": "Variável alvo para classificação.",
    },
}

PAY_STATUS_LEGEND = (
    "Legenda PAY_*: -2 sem consumo | -1 pago em dia | 0 pagamento adiantado | "
    "1..9 atraso (quantidade de meses)."
)


def column_dictionary_rows() -> list[dict[str, str]]:
    """Dicionário embutido do dataset (descrição + sugestão de uso)."""
    rows: list[dict[str, str]] = []
    for col, meta in COLUMN_DOCS.items():
        if col == TARGET_COLUMN:
            sugestao = "alvo"
        elif col in SUGGESTED_DROP_COLUMNS:
            sugestao = "remover_recomendado"
        elif col in OPTIONAL_DROP_COLUMNS:
            sugestao = "remover_opcional"
        else:
            sugestao = "manter"
        rows.append(
            {
                "coluna": col,
                "tipo": meta.get("tipo", ""),
                "descricao": meta.get("descricao", ""),
                "notas": meta.get("notas", ""),
                "sugestao": sugestao,
                "motivo_remocao": DROP_REASONS.get(col, ""),
            }
        )
    return rows
