import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from chat_store import DEFAULT_CHAT_TITLE, create_chat, load_chats, save_chats
from config_loader import (
    FlowiseSettings,
    GeminiSettings,
    LmStudioSettings,
    get_chat_backend,
    get_flowise_settings,
    get_lmstudio_settings,
    get_settings,
)
from flowise_client import FlowiseAPIError, extract_reply as flowise_extract_reply, query as flowise_query
from lmstudio_client import LmStudioError, chat as lmstudio_chat, check_server as lmstudio_check
from duckdb_store import ensure_duckdb_database, get_datasets, get_duckdb_path, get_sample_queries, preview_table, run_query
from text_utils import repair_mojibake
import os
from langfuse import get_client

TXTAI_AVAILABLE = True
try:
    from txtai_store import (
        ensure_txtai_index,
        get_txtai_status,
        get_store_documents,
        has_gemini_api_key,
        is_index_ready,
        peek_index,
        query_index,
        rebuild_index,
    )
except ImportError:
    TXTAI_AVAILABLE = False

    def has_gemini_api_key() -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def is_index_ready() -> bool:
        return False

    def get_txtai_status() -> dict:
        return {
            "path": "n/d",
            "collection_name": "n/d",
            "document_files": 0,
            "collections": [],
            "chunk_count": 0,
            "embedding_model": "nao instalado",
            "has_openai_api_key": False,
        }

    def get_store_documents():
        from dataclasses import dataclass
        from pathlib import Path

        @dataclass(frozen=True)
        class _Doc:
            file_name: str
            path: Path
            title: str

        docs_dir = Path(__file__).parent / "store_docs"
        if not docs_dir.is_dir():
            return []
        return [
            _Doc(file_name=p.name, path=p, title=p.stem.replace("_", " ").title())
            for p in sorted(docs_dir.glob("*.md"))
        ]

    def query_index(*_args, **_kwargs):
        return []

    def peek_index(*_args, **_kwargs):
        return []

    def rebuild_index():
        raise RuntimeError("txtai nao instalado nesta imagem.")

    def ensure_txtai_index():
        return None

genai = None
try:
    import google.generativeai as genai
except ImportError:
    pass

STREAMING_DELAY_SECONDS = 0.03
MAX_CHAT_TITLE_LENGTH = 40
EMPTY_ASSISTANT_REPLY = "O assistente nao retornou texto nesta execucao."
SUGGESTED_CHAT_QUESTIONS = (
    "Quais iPhones custam ate R$ 7.000?",
    "Qual regiao tem mais estoque do iPhone 16?",
    "Qual foi o prazo medio de entrega no Sudeste?",
)

st.set_page_config(page_title="Chat Loja iPhone - Flowise + txtai", layout="wide")


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_timestamp(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp)
        return parsed.astimezone().strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return timestamp


def is_chat_available() -> bool:
    backend = get_chat_backend()
    if backend == "flowise":
        try:
            get_flowise_settings()
            return True
        except ValueError:
            return False
    if backend == "lmstudio":
        try:
            get_lmstudio_settings()
            return True
        except ValueError:
            return False
    return has_gemini_api_key()


def ensure_state() -> None:
    if "txtai_ready" not in st.session_state:
        st.session_state.txtai_ready = is_index_ready()

    if "chats" not in st.session_state:
        chats = load_chats()
        if not chats:
            chats = [create_chat()]
        save_chats(chats)
        st.session_state.chats = chats

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]

    if not any(chat["id"] == st.session_state.active_chat_id for chat in st.session_state.chats):
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]

    sample_queries = get_sample_queries()
    if "selected_demo_query" not in st.session_state:
        st.session_state.selected_demo_query = sample_queries[0][0]
    if "demo_sql" not in st.session_state:
        st.session_state.demo_sql = sample_queries[0][1]
    if "chroma_query" not in st.session_state:
        st.session_state.chroma_query = "Como funciona o prazo de entrega?"


def persist_chats() -> None:
    st.session_state.chats.sort(key=lambda chat: chat["updated_at"], reverse=True)
    save_chats(st.session_state.chats)


def get_active_chat() -> dict:
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat

    fallback_chat = create_chat()
    st.session_state.chats.insert(0, fallback_chat)
    st.session_state.active_chat_id = fallback_chat["id"]
    persist_chats()
    return fallback_chat


def build_chat_title(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user":
            title = " ".join(message["content"].split())
            if not title:
                break
            if len(title) > MAX_CHAT_TITLE_LENGTH:
                return f"{title[: MAX_CHAT_TITLE_LENGTH - 3]}..."
            return title
    return DEFAULT_CHAT_TITLE


def update_chat_metadata(chat: dict) -> None:
    chat["title"] = build_chat_title(chat["messages"])
    chat["updated_at"] = current_timestamp()


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def start_new_chat() -> None:
    new_chat = create_chat()
    st.session_state.chats.insert(0, new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    persist_chats()


def clear_active_chat() -> None:
    chat = get_active_chat()
    chat["title"] = DEFAULT_CHAT_TITLE
    chat["messages"] = []
    chat["flowise_chat_id"] = None
    chat["updated_at"] = current_timestamp()
    persist_chats()


def apply_selected_demo_query() -> None:
    selected_name = st.session_state.selected_demo_query
    for query_name, sql in get_sample_queries():
        if query_name == selected_name:
            st.session_state.demo_sql = sql
            return


def stream_text(text: str):
    tokens = re.findall(r"\S+|\s+", text)
    for token in tokens:
        yield token
        if token.strip():
            time.sleep(STREAMING_DELAY_SECONDS)


@st.cache_data(show_spinner=False)
def load_table_preview(table_name: str, limit: int) -> pd.DataFrame:
    return preview_table(table_name, limit)


@st.cache_data(show_spinner=False)
def execute_duckdb_query(sql: str) -> pd.DataFrame:
    return run_query(sql)


def render_sidebar(
    settings: GeminiSettings | None,
    flowise_settings: FlowiseSettings | None,
    lmstudio_settings: LmStudioSettings | None,
    database_path: Path,
    app_error: str | None,
) -> None:
    with st.sidebar:
        st.header("Conversas")
        st.button("Novo chat", on_click=start_new_chat, width="stretch", type="primary")
        st.button("Limpar chat atual", on_click=clear_active_chat, width="stretch")
        st.caption(f"{len(st.session_state.chats)} chats salvos")

        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            st.button(
                chat["title"],
                key=f"chat-{chat['id']}",
                on_click=set_active_chat,
                args=(chat["id"],),
                width="stretch",
                type="primary" if is_active else "secondary",
                help=f"Atualizado em {format_timestamp(chat['updated_at'])}",
            )

        with st.expander("Status da aplicacao", expanded=True):
            st.caption(f"Chat: **{get_chat_backend()}**")
            if lmstudio_settings:
                ok, msg = lmstudio_check()
                if ok:
                    st.success(f"LM Studio OK — modelo `{lmstudio_settings.model}`")
                else:
                    st.error(f"LM Studio offline: {msg}")
                st.caption(lmstudio_settings.base_url)
            elif flowise_settings:
                st.success("Flowise configurado (LLM do flow)")
                st.caption(flowise_settings.api_url)
            elif get_chat_backend() == "flowise":
                st.error(app_error or "Flowise nao configurado.")
            elif get_chat_backend() == "lmstudio":
                st.error(app_error or "LM Studio nao configurado.")
            if settings and has_gemini_api_key():
                st.caption(f"Gemini embeddings: {settings.embedding_model}")
            st.info(f"DuckDB pronto em `{database_path}`")
            st.info(f"txtai local em `{Path('data/txtai')}`")


def render_chat_intro(active_chat: dict, *, enable_suggestions: bool) -> str | None:
    st.subheader(active_chat["title"])
    info_col, stats_col = st.columns([2, 1])
    with info_col:
        backend = get_chat_backend()
        if backend == "lmstudio":
            st.caption("Chat direto com LM Studio (sem Agent do Flowise).")
        elif backend == "flowise":
            st.caption("Chat via Flowise. RAG txtai e opcional.")
        else:
            st.caption("Chat via Gemini com RAG txtai opcional.")
    with stats_col:
        st.metric("Mensagens", len(active_chat["messages"]))

    if active_chat["messages"] or not enable_suggestions:
        return None

    st.info("Comece uma conversa ou use uma das perguntas guiadas abaixo.")
    question_columns = st.columns(len(SUGGESTED_CHAT_QUESTIONS))
    selected_prompt = None
    for column, question in zip(question_columns, SUGGESTED_CHAT_QUESTIONS):
        if column.button(question, key=f"suggested-{question}", width="stretch"):
            selected_prompt = question
    return selected_prompt


def render_chat_history(active_chat: dict) -> None:
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(repair_mojibake(message["content"]))


def get_rag_context(prompt: str) -> str:
    if not TXTAI_AVAILABLE or not is_index_ready() or not has_gemini_api_key():
        return ""

    try:
        context_chunks = query_index(prompt, n_results=3)
    except Exception:
        return ""

    if not context_chunks:
        return ""

    return "\n\n".join([
        f"Documento: {chunk['title']} (Arquivo: {chunk['source_file']})\n{repair_mojibake(chunk['text'])}"
        for chunk in context_chunks
    ])


def get_rag_context_and_prompt(prompt: str) -> tuple[str, str]:
    context_str = get_rag_context(prompt)
    system_content = (
        "Você é um assistente virtual inteligente de uma loja de iPhone.\n"
        "Use as informações dos documentos abaixo para responder à pergunta do cliente de forma precisa, amigável e profissional.\n"
        "Se a informação não estiver disponível nos documentos fornecidos ou no banco de dados, "
        "diga que não tem essa informação no momento, mas que o cliente pode entrar em contato com o pós-venda.\n\n"
        "Documentos de Contexto:\n"
        f"{context_str or 'Nenhum documento relevante encontrado.'}"
    )
    return context_str, system_content


def handle_chat_prompt_lmstudio(prompt: str, active_chat: dict) -> None:
    active_chat["messages"].append({"role": "user", "content": prompt})
    update_chat_metadata(active_chat)
    persist_chats()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            context_str = get_rag_context(prompt)
            if context_str:
                with st.expander("Contexto recuperado (txtai)", expanded=False):
                    st.markdown(context_str)

            system_prompt = (
                "Voce e um assistente de uma loja de iPhone. "
                "Responda de forma clara e profissional em portugues."
            )
            if context_str:
                system_prompt += f"\n\nContexto da loja:\n{context_str}"

            history = active_chat["messages"][:-1][-10:]
            with st.spinner("Consultando LM Studio..."):
                reply = lmstudio_chat(
                    prompt,
                    history=history,
                    system_prompt=system_prompt,
                )
            if not reply.strip():
                reply = EMPTY_ASSISTANT_REPLY
            st.markdown(reply)
        except LmStudioError as exc:
            reply = f"**Erro LM Studio:** {exc}"
            st.error(str(exc))
            st.markdown(reply)
        except Exception as exc:
            reply = f"Erro ao processar resposta: {exc}"
            st.error(reply)

    active_chat["messages"].append({"role": "assistant", "content": reply})
    update_chat_metadata(active_chat)
    persist_chats()


def handle_chat_prompt_flowise(prompt: str, active_chat: dict, flowise_settings: FlowiseSettings) -> None:
    active_chat["messages"].append({"role": "user", "content": prompt})
    update_chat_metadata(active_chat)
    persist_chats()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            context_str = get_rag_context(prompt)
            if context_str:
                with st.expander("Contexto recuperado (txtai)", expanded=False):
                    st.markdown(context_str)

            question = prompt
            if context_str:
                question = (
                    "Use o contexto abaixo para responder ao cliente.\n\n"
                    f"{context_str}\n\n"
                    f"Pergunta do cliente: {prompt}"
                )

            payload: dict = {"question": question}
            if active_chat.get("flowise_chat_id"):
                payload["chatId"] = active_chat["flowise_chat_id"]

            with st.spinner("Consultando Flowise..."):
                result = flowise_query(payload)

            reply = flowise_extract_reply(result)
            if result.get("chatId"):
                active_chat["flowise_chat_id"] = result["chatId"]

            st.markdown(reply)
        except FlowiseAPIError as exc:
            reply = f"**Erro do Flowise:** {exc}"
            err_text = str(exc)
            if "Unauthorized" in err_text:
                st.error("Token invalido em config.toml → [flowise] bearer_token")
            elif "Connection error" in err_text or "Agent node" in err_text:
                st.warning("Flowise Agent falhou — tentando LM Studio direto...")
                try:
                    context_str = get_rag_context(prompt)
                    system_prompt = (
                        "Voce e um assistente de uma loja de iPhone. "
                        "Responda em portugues de forma clara."
                    )
                    if context_str:
                        system_prompt += f"\n\nContexto:\n{context_str}"
                    reply = lmstudio_chat(
                        prompt,
                        history=active_chat["messages"][:-1][-10:],
                        system_prompt=system_prompt,
                    )
                    st.markdown(reply or EMPTY_ASSISTANT_REPLY)
                    active_chat["messages"].append(
                        {"role": "assistant", "content": reply or EMPTY_ASSISTANT_REPLY}
                    )
                    update_chat_metadata(active_chat)
                    persist_chats()
                    return
                except LmStudioError as lm_exc:
                    reply = f"**Erro LM Studio:** {lm_exc}"
                    st.error(str(lm_exc))
            else:
                st.error(err_text)
            st.markdown(reply)
        except Exception as exc:
            reply = f"Erro ao processar resposta: {exc}"
            st.error(reply)

    active_chat["messages"].append({"role": "assistant", "content": reply})
    update_chat_metadata(active_chat)
    persist_chats()


def handle_chat_prompt_gemini(prompt: str, active_chat: dict, settings: GeminiSettings) -> None:
    active_chat["messages"].append({"role": "user", "content": prompt})
    update_chat_metadata(active_chat)
    persist_chats()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            context_str, system_content = get_rag_context_and_prompt(prompt)
            
            if context_str:
                with st.expander("Contexto recuperado (RAG)", expanded=False):
                    st.markdown(context_str)
            
            contents = []
            for msg in active_chat["messages"][:-1][-10:]:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})
            contents.append({"role": "user", "parts": [prompt]})

            if genai is None:
                raise RuntimeError("google-generativeai nao instalado. Use backend flowise ou imagem Docker full.")

            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            if gemini_key:
                genai.configure(api_key=gemini_key)

            model = genai.GenerativeModel(
                model_name=settings.model,
                system_instruction=system_content
            )
            
            def gemini_stream():
                response = model.generate_content(contents, stream=True)
                for chunk in response:
                    try:
                        text = chunk.text
                        if text:
                            yield text
                    except (ValueError, AttributeError):
                        pass

            reply = repair_mojibake(st.write_stream(gemini_stream()) or "")
            if not reply or not reply.strip():
                reply = EMPTY_ASSISTANT_REPLY
                st.markdown(reply)
                
        except Exception as exc:
            reply = f"Erro ao processar resposta: {exc}"
            st.error(reply)

    active_chat["messages"].append({"role": "assistant", "content": reply})
    update_chat_metadata(active_chat)
    persist_chats()


def handle_chat_prompt(prompt: str, active_chat: dict, settings: GeminiSettings | None) -> None:
    backend = get_chat_backend()
    if backend == "lmstudio":
        handle_chat_prompt_lmstudio(prompt, active_chat)
        return
    if backend == "flowise":
        handle_chat_prompt_flowise(prompt, active_chat, get_flowise_settings())
        return
    if not settings:
        st.error("Gemini nao configurado para o chat.")
        return
    handle_chat_prompt_gemini(prompt, active_chat, settings)


def render_query_metrics(dataframe: pd.DataFrame) -> None:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    metric_columns = st.columns(3)
    metric_columns[0].metric("Linhas", len(dataframe))
    metric_columns[1].metric("Colunas", len(dataframe.columns))
    if numeric_columns:
        metric_columns[2].metric("Soma de referencia", f"{dataframe[numeric_columns[0]].sum():,.0f}")
    else:
        metric_columns[2].metric("Valores unicos", dataframe.nunique().sum())


def render_duckdb_demo(database_path: Path) -> None:
    st.subheader("DuckDB + pandas")
    st.caption("As consultas sao executadas no DuckDB e o resultado chega ao app como pandas DataFrame.")

    summary_columns = st.columns(3)
    summary_columns[0].metric("Arquivos demo", len(get_datasets()))
    summary_columns[1].metric("Consultas exemplo", len(get_sample_queries()))
    summary_columns[2].metric("Banco local", database_path.name)

    st.selectbox(
        "Consulta exemplo",
        options=[query_name for query_name, _ in get_sample_queries()],
        key="selected_demo_query",
        on_change=apply_selected_demo_query,
    )
    st.text_area("SQL", key="demo_sql", height=220)

    if st.button("Executar consulta", width="stretch"):
        try:
            query_result = execute_duckdb_query(st.session_state.demo_sql)
            st.success("Consulta executada com sucesso.")
            render_query_metrics(query_result)
            st.dataframe(query_result, width="stretch", hide_index=True)

            label_columns = query_result.select_dtypes(exclude="number").columns.tolist()
            numeric_columns = query_result.select_dtypes(include="number").columns.tolist()
            if label_columns and numeric_columns:
                chart_frame = query_result[[label_columns[0], numeric_columns[0]]].dropna().head(10).set_index(label_columns[0])
                st.bar_chart(chart_frame)
        except Exception as exc:
            st.error(f"Erro ao executar consulta no DuckDB: {exc}")

    st.divider()
    st.markdown("#### Demonstracao: DuckDB alimentando pandas")
    st.caption("Primeiro consultamos o DuckDB. Depois usamos pandas para agrupar e resumir os dados no Streamlit.")

    orders_frame = execute_duckdb_query(
        """
SELECT
    p.data_pedido,
    p.regiao_entrega,
    p.status_pedido,
    p.prazo_prometido_dias,
    p.prazo_real_dias,
    p.valor_total_brl,
    c.modelo
FROM pedidos_clientes p
JOIN catalogo_iphones c ON c.sku = p.sku
        """.strip()
    )

    delivered_orders = orders_frame[orders_frame["status_pedido"] == "entregue"].copy()
    sales_by_model = (
        orders_frame[orders_frame["status_pedido"] != "cancelado"]
        .groupby("modelo", as_index=False)
        .agg(total_pedidos=("modelo", "size"), receita_brl=("valor_total_brl", "sum"))
        .sort_values("receita_brl", ascending=False)
    )
    delivery_by_region = (
        delivered_orders.groupby("regiao_entrega", as_index=False)
        .agg(
            prazo_prometido_medio=("prazo_prometido_dias", "mean"),
            prazo_real_medio=("prazo_real_dias", "mean"),
            pedidos_entregues=("regiao_entrega", "size"),
        )
        .sort_values("prazo_real_medio", ascending=True)
    )

    report_columns = st.columns(2)
    with report_columns[0]:
        st.markdown("**Receita por modelo**")
        st.dataframe(sales_by_model, width="stretch", hide_index=True)
        st.bar_chart(sales_by_model.set_index("modelo")["receita_brl"])
    with report_columns[1]:
        st.markdown("**Prazo medio por regiao**")
        st.dataframe(delivery_by_region, width="stretch", hide_index=True)
        st.bar_chart(delivery_by_region.set_index("regiao_entrega")["prazo_real_medio"])


def render_demo_files() -> None:
    st.subheader("Arquivos demo")
    st.caption("Tres arquivos ficticios para testes de chat, consultas SQL e relatorios.")

    for dataset in get_datasets():
        with st.expander(f"{dataset.title} - {dataset.file_name}", expanded=True):
            info_columns = st.columns([2, 1])
            with info_columns[0]:
                st.write(dataset.description)
            with info_columns[1]:
                total_rows = execute_duckdb_query(f"SELECT COUNT(*) AS total FROM {dataset.table_name}").iloc[0]["total"]
                st.metric("Linhas", int(total_rows))

            preview_frame = load_table_preview(dataset.table_name, 8)
            st.dataframe(preview_frame, width="stretch", hide_index=True)
            st.download_button(
                label=f"Baixar {dataset.file_name}",
                data=dataset.file_path.read_bytes(),
                file_name=dataset.file_name,
                mime="text/csv",
                key=f"download-{dataset.table_name}",
            )

    st.divider()
    st.markdown("#### Documentos da loja para o ChromaDB")
    st.caption("Base textual usada para testes de embeddings, busca semantica e validacao do Chroma no proprio container.")

    for store_document in get_store_documents():
        with st.expander(store_document.title, expanded=False):
            st.caption(store_document.file_name)
            content = store_document.path.read_text(encoding="utf-8")
            st.markdown(content)
            st.download_button(
                label=f"Baixar {store_document.file_name}",
                data=store_document.path.read_bytes(),
                file_name=store_document.file_name,
                mime="text/markdown",
                key=f"download-doc-{store_document.file_name}",
            )


def render_txtai_manager() -> None:
    st.subheader("txtai")
    st.caption("Gerencie o indice vetorial local da loja e faca consultas rapidas sem sair do Streamlit.")

    if not TXTAI_AVAILABLE:
        st.warning(
            "Modulo txtai nao esta nesta imagem (build leve). "
            "Use `start-light.ps1` so para Flowise, ou build completo apos liberar o disco do Docker."
        )
        return

    status = get_txtai_status()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Arquivos markdown", status["document_files"])
    metric_columns[1].metric("Colecoes", len(status["collections"]))
    metric_columns[2].metric("Chunks indexados", status["chunk_count"])
    metric_columns[3].metric("Embedding model", status["embedding_model"])

    info_columns = st.columns([2, 1])
    with info_columns[0]:
        st.code(
            "\n".join(
                [
                    f"path: {status['path']}",
                    f"collection: {status['collection_name']}",
                    f"api_key_configurada: {status['has_openai_api_key']}",
                ]
            ),
            language="text",
        )
    with info_columns[1]:
        st.markdown("**Operacoes**")
        if st.button("Reindexar documentos", width="stretch", type="primary"):
            try:
                rebuild_result = rebuild_index()
                st.success(
                    f"Indice `{rebuild_result['collection_name']}` atualizado com "
                    f"{rebuild_result['files_indexed']} arquivos e {rebuild_result['chunks_indexed']} chunks."
                )
            except Exception as exc:
                st.error(f"Erro ao reindexar no txtai: {exc}")

    if not has_gemini_api_key():
        st.warning(
            "GEMINI_API_KEY nao configurada. Sem ela, voce consegue ver os documentos e o status do txtai, "
            "mas nao consegue reindexar ou fazer busca semantica."
        )
    else:
        st.success(
            "A chave do Gemini esta sendo lida por ambiente. Nenhuma API key foi gravada no codigo."
        )

    st.divider()
    st.markdown("#### Documentos preparados para indexacao")
    docs_frame = pd.DataFrame(
        [{"arquivo": doc.file_name, "titulo": doc.title, "caminho": str(doc.path)} for doc in get_store_documents()]
    )
    st.dataframe(docs_frame, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Consulta semantica rapida")
    query_col, count_col = st.columns([3, 1])
    with query_col:
        query_text = st.text_input("Pergunta", key="chroma_query")
    with count_col:
        top_k = st.slider("Top K", min_value=1, max_value=8, value=4)

    if st.button("Pesquisar no txtai", width="stretch"):
        try:
            results = query_index(query_text, n_results=top_k)

            if not results:
                st.info("Nenhum trecho encontrado para essa consulta.")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r.get('title', 'Documento sem titulo')}**")
                    st.caption(
                        f"arquivo: {r.get('source_file', 'n/d')} | "
                        f"chunk: {r.get('chunk_index', 'n/d')} | "
                        f"score (similaridade): {r.get('score', 0.0):.4f}"
                    )
                    st.write(r.get('text', ''))
        except Exception as exc:
            st.error(f"Erro na consulta semantica: {exc}")

    st.divider()
    st.markdown("#### Peek do indice (Visualizacao via SQL)")
    try:
        peek = peek_index(limit=6)
        peek_rows = []
        for r in peek:
            peek_rows.append(
                {
                    "id": r.get("id", ""),
                    "arquivo": r.get("source_file", ""),
                    "titulo": r.get("title", ""),
                    "chunk": r.get("chunk_index", ""),
                    "trecho": r.get("text", "")[:180],
                }
            )
        if peek_rows:
            st.dataframe(pd.DataFrame(peek_rows), width="stretch", hide_index=True)
        else:
            st.info("Indice existente, mas sem chunks visiveis no peek.")
    except Exception as exc:
        st.info(f"Peek indisponivel no momento: {exc}")


ensure_state()
database_path = ensure_duckdb_database()

app_settings: GeminiSettings | None = None
flowise_settings: FlowiseSettings | None = None
lmstudio_settings: LmStudioSettings | None = None
app_error: str | None = None
try:
    backend = get_chat_backend()
    if backend == "flowise":
        flowise_settings = get_flowise_settings()
    elif backend == "lmstudio":
        lmstudio_settings = get_lmstudio_settings()
    else:
        app_settings = get_settings()
except ValueError as exc:
    app_error = str(exc)

if has_gemini_api_key():
    try:
        app_settings = get_settings()
    except ValueError:
        pass

render_sidebar(app_settings, flowise_settings, lmstudio_settings, get_duckdb_path(), app_error)

st.title("Loja iPhone — Chat, dados e ML")
st.caption("Chat via Flowise · DuckDB · txtai · ML tradicional (FLAML + UCI 350).")

chat_tab, duckdb_tab, txtai_tab, ml_tab, files_tab = st.tabs(
    ["Chat", "DuckDB + pandas", "txtai", "ML tradicional", "Arquivos demo"]
)
active_chat_state = get_active_chat()

with chat_tab:
    enable_suggestions = is_chat_available()
    selected_question = render_chat_intro(active_chat_state, enable_suggestions=enable_suggestions)
    render_chat_history(active_chat_state)
    if not is_chat_available():
        backend = get_chat_backend()
        if backend == "lmstudio":
            st.warning("LM Studio nao configurado ou offline.")
            st.info("Abra LM Studio, carregue um modelo, Start Server (porta 1234), depois reinicie: SUBIR.cmd")
        elif backend == "flowise":
            st.warning("Flowise nao configurado. Preencha [flowise] em config.toml e reinicie o container.")
            st.info("Ou mude chat_backend = 'lmstudio' em config.toml (padrao corrigido).")
        else:
            st.warning("GEMINI_API_KEY nao configurada para o chat.")
    else:
        prompt_from_input = st.chat_input("Digite sua mensagem para o assistente...")
        prompt_to_process = selected_question or prompt_from_input
        if prompt_to_process:
            handle_chat_prompt(prompt_to_process, active_chat_state, app_settings)

with duckdb_tab:
    render_duckdb_demo(database_path)

with txtai_tab:
    render_txtai_manager()

with ml_tab:
    try:
        from ml_traditional.ui import render_ml_tab

        render_ml_tab()
    except OSError as exc:
        if "libgomp" in str(exc):
            st.error(
                "Biblioteca do sistema faltando (libgomp). Reconstrua a imagem: "
                "feche o container e rode **SUBIR.cmd** de novo."
            )
        else:
            st.error(f"Erro ao carregar ML tradicional: {exc}")
    except ImportError as exc:
        st.error(f"Dependencias de ML nao instaladas: {exc}")

with files_tab:
    render_demo_files()
