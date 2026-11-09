# Assistente Inteligente para Loja de iPhones

Aplicação desenvolvida em Streamlit para simular o atendimento de uma loja fictícia de iPhones, utilizando inteligência artificial, agentes especializados, consulta a dados e Machine Learning.

## Sobre o projeto

O projeto tem como objetivo demonstrar como tecnologias de inteligência artificial podem apoiar o atendimento ao cliente e auxiliar análises relacionadas ao negócio.

A aplicação possui:
- Chat com agentes de IA para atendimento geral e vendas de iPhones.
- Integração com Flowise para organização do fluxo de agentes.
- Utilização de LM Studio para execução local de modelos de linguagem.
- Armazenamento e consulta de dados com DuckDB.
- Módulo independente de Machine Learning para classificação de inadimplência.
- Treinamento de modelos com FLAML AutoML.
- Golden Dataset e testes automatizados utilizando unittest.

## Arquitetura

O sistema é organizado em módulos independentes:

- `app.py`: aplicação principal e interface de atendimento.
- `ml_app.py`: módulo de Machine Learning tradicional.
- `flowise_client.py`: integração com Flowise.
- `lmstudio_client.py`: comunicação com LM Studio.
- `duckdb_store.py`: armazenamento e consulta de dados.
- `chat_store.py`: gerenciamento do histórico de conversas.
- `config_loader.py`: carregamento de configurações.
- `tests/`: conjunto de perguntas de referência e testes automatizados.

## Tecnologias utilizadas

- Python
- Streamlit
- Flowise
- LM Studio
- DuckDB
- FLAML AutoML
- Docker
- unittest

## Como executar

### Pré-requisitos

- Python instalado.
- Docker Desktop, caso deseje executar a aplicação em contêineres.
- LM Studio configurado para disponibilizar o modelo local.
- Flowise configurado com o fluxo de atendimento.

### Execução local

1. Clone o repositório:

   ```bash
   git clone https://github.com/vittinxl/assistente-iphone.git
   cd assistente-iphone
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual no Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Inicie a aplicação:

   ```bash
   streamlit run app.py
   ```

6. Acesse:

   ```text
   http://localhost:8501
   ```

### Execução com Docker

Com o Docker Desktop instalado e iniciado, execute:

```bash
docker compose up --build
```

A aplicação ficará disponível em:

```text
http://localhost:8501
```

No Windows, também podem ser utilizados os arquivos auxiliares de inicialização disponíveis no repositório.

## Machine Learning

O módulo de Machine Learning utiliza o dataset **Default of Credit Card Clients**, disponibilizado pela UCI Machine Learning Repository.

- Instâncias originais: 30.000.
- Características: 23.
- Tarefa: classificação de inadimplência.
- Ferramenta de treinamento: FLAML AutoML.
- Estimador observado: Random Forest.

Os resultados obtidos apresentaram limitações na identificação da classe positiva, indicando a necessidade de melhorias no treinamento e na avaliação do modelo.

## Testes

O projeto possui um Golden Dataset com dez perguntas de referência relacionadas ao atendimento da loja.

Para executar os testes automatizados:

```bash
python -m unittest discover -s tests -v
```

Os testes verificam a quantidade de perguntas, os campos obrigatórios, o preenchimento dos dados e a unicidade dos identificadores.

## Limitações conhecidas

- A integração com Langfuse não foi concluída.
- O módulo de Machine Learning funciona separadamente do chat.
- O classificador de inadimplência apresentou desempenho limitado na identificação dos casos positivos.
- Os testes automatizados verificam a estrutura do Golden Dataset, não a qualidade semântica das respostas.
- A avaliação individual das respostas e a observabilidade dos agentes são melhorias futuras.

## Melhorias futuras

- Implementar observabilidade com Langfuse.
- Melhorar o desempenho do classificador de inadimplência.
- Ampliar os testes de qualidade das respostas dos agentes.
- Integrar análises de risco a funcionalidades comerciais.
- Aprimorar a experiência de atendimento.
