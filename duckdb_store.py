import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


APP_DIR = Path(__file__).parent
SAMPLE_DATA_DIR = APP_DIR / "sample_data"
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", APP_DIR / "data" / "loja_iphone.duckdb"))


@dataclass(frozen=True)
class DemoDataset:
    table_name: str
    file_name: str
    title: str
    description: str

    @property
    def file_path(self) -> Path:
        return SAMPLE_DATA_DIR / self.file_name


DATASETS = (
    DemoDataset(
        table_name="catalogo_iphones",
        file_name="catalogo_iphones.csv",
        title="Catalogo de iPhones",
        description="Modelos, cores, armazenamento, preco e atributos comerciais.",
    ),
    DemoDataset(
        table_name="estoque_lojas",
        file_name="estoque_lojas.csv",
        title="Estoque por canal e regiao",
        description="Disponibilidade por cidade, canal de venda e previsao de reposicao.",
    ),
    DemoDataset(
        table_name="pedidos_clientes",
        file_name="pedidos_clientes.csv",
        title="Pedidos recentes",
        description="Pedidos ficticios com status, prazo e forma de pagamento para analise.",
    ),
)


SAMPLE_QUERIES = (
    (
        "Modelos ate R$ 7.000",
        """
SELECT modelo, armazenamento_gb, cor, preco_brl, parcelamento_sem_juros
FROM catalogo_iphones
WHERE preco_brl <= 7000
ORDER BY preco_brl ASC, modelo ASC
        """.strip(),
    ),
    (
        "Estoque do Sudeste por modelo",
        """
SELECT c.modelo, e.estado, e.cidade, e.estoque_disponivel, e.retirada_hoje
FROM estoque_lojas e
JOIN catalogo_iphones c ON c.sku = e.sku
WHERE e.regiao = 'Sudeste'
ORDER BY e.estoque_disponivel DESC, c.modelo ASC
        """.strip(),
    ),
    (
        "Receita e pedidos por modelo",
        """
SELECT c.modelo, COUNT(*) AS total_pedidos, SUM(p.valor_total_brl) AS receita_brl
FROM pedidos_clientes p
JOIN catalogo_iphones c ON c.sku = p.sku
WHERE p.status_pedido <> 'cancelado'
GROUP BY c.modelo
ORDER BY receita_brl DESC
        """.strip(),
    ),
    (
        "Prazo medio de entrega por regiao",
        """
SELECT
    regiao_entrega,
    ROUND(AVG(prazo_prometido_dias), 2) AS prazo_prometido_medio,
    ROUND(AVG(prazo_real_dias), 2) AS prazo_real_medio,
    SUM(CASE WHEN prazo_real_dias > prazo_prometido_dias THEN 1 ELSE 0 END) AS pedidos_atrasados
FROM pedidos_clientes
WHERE status_pedido = 'entregue'
GROUP BY regiao_entrega
ORDER BY prazo_real_medio ASC
        """.strip(),
    ),
)


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def get_datasets() -> tuple[DemoDataset, ...]:
    return DATASETS


def get_sample_queries() -> tuple[tuple[str, str], ...]:
    return SAMPLE_QUERIES


def get_duckdb_path() -> Path:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DUCKDB_PATH


def ensure_duckdb_database() -> Path:
    database_path = get_duckdb_path()
    connection = duckdb.connect(str(database_path))
    try:
        for dataset in DATASETS:
            connection.execute(
                f"""
CREATE OR REPLACE VIEW {dataset.table_name} AS
SELECT *
FROM read_csv_auto('{_sql_path(dataset.file_path)}', HEADER = TRUE)
                """
            )
    finally:
        connection.close()

    return database_path


def run_query(sql: str) -> pd.DataFrame:
    ensure_duckdb_database()
    connection = duckdb.connect(str(get_duckdb_path()))
    try:
        return connection.execute(sql).df()
    finally:
        connection.close()


def preview_table(table_name: str, limit: int = 10) -> pd.DataFrame:
    safe_limit = max(1, min(limit, 100))
    return run_query(f"SELECT * FROM {table_name} LIMIT {safe_limit}")
