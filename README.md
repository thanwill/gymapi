# GymAPI

GymAPI é uma aplicação Django para gerenciar datasets de academias, realizar análises e previsões com base nesses dados.

## Funcionalidades

- Upload de datasets em formato CSV.
- Processamento e análise de datasets.
- Treinamento de modelos de machine learning.
- Armazenamento de metadados das colunas dos datasets.
- Geração de relatórios de análise e previsões.
- Remoção de análises.

## Requisitos

- Python 3.11
- Django 5.1.3
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Django REST Framework

## Instalação

1. Clone o repositório:

   ```sh
   git clone https://github.com/seu-usuario/gymapi.git
   cd gymapi

   Claro! Aqui está um exemplo de README para o seu projeto:


# GymAPI

GymAPI é uma aplicação Django para gerenciar datasets de academias, realizar análises e previsões com base nesses dados.

## Funcionalidades

- Upload de datasets em formato CSV.
- Processamento e análise de datasets.
- Treinamento de modelos de machine learning.
- Armazenamento de metadados das colunas dos datasets.
- Geração de relatórios de análise e previsões.
- Remoção de análises.

## Requisitos

- Python 3.11
- Django 5.1.3
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Django REST Framework

## Instalação

1. Clone o repositório:

   ```sh
   git clone https://github.com/seu-usuario/gymapi.git
   cd gymapi
   ```

2. Crie um ambiente virtual e ative-o:

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # No Windows, use .venv\Scripts\activate
   ```

3. Instale as dependências:

   ```sh
   pip install -r requirements.txt
   ```

4. Realize as migrações do banco de dados:

   ```sh
   python manage.py makemigrations
   python manage.py migrate
   ```

5. Inicie o servidor de desenvolvimento:

   ```sh
   python manage.py runserver
   ```

## Uso

### Upload de Dataset

Endpoint: `POST /api/datasets/upload`

Envie um arquivo CSV para o endpoint para fazer o upload do dataset.

### Listar Datasets

Endpoint: `GET /api/datasets`

Retorna uma lista de todos os datasets.

### Criar Análise

Endpoint: `POST /api/analyses/<dataset_id>`

Cria uma análise para o dataset especificado.

### Remover Análise

Endpoint: `DELETE /api/analyses/<analysis_id>/delete`

Remove a análise especificada.

### Obter Resultados de Análises

Endpoint: `GET /api/analyses`

Retorna todos os resultados de análises.

### Obter Correlações

Endpoint: `GET /api/analyses/<dataset_id>/correlations`

Retorna as correlações entre as colunas do dataset especificado.

## Estrutura do Projeto

- `api/`: Contém as views, serializers e utils.
- `gymdatabase/`: Configurações do projeto Django.
- `models.py`: Definição dos modelos do banco de dados.
- `views.py`: Definição das views da API.
- `urls.py`: Definição das rotas da API.
- `utils.py`: Funções utilitárias para processamento de datasets e análises.

## Contribuição

1. Faça um fork do projeto.
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`).
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`).
4. Faça um push para a branch (`git push origin feature/nova-feature`).
5. Crie um novo Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
```

Este README fornece uma visão geral do projeto, instruções de instalação, uso dos endpoints da API, estrutura do projeto e informações sobre como contribuir. Ajuste conforme necessário para refletir as especificidades do seu projeto.
