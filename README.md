# LLM's UI Apps

Este é um aplicativo Streamlit que possui diferentes funcionalidades. Antes de executar o aplicativo, é necessário definir a variável de ambiente **OPENAI_API_KEY** com a chave de API da OpenAI. Essa chave é necessária para que o aplicativo possa utilizar os modelos e serviços da OpenAI. Certifique-se de ter uma chave de API válida antes de prosseguir.

##  Criador de script para Youtube
Nesta opção, o aplicativo gera um script para um vídeo no Youtube com base em um determinado tema. O usuário fornece um tema e o aplicativo gera um título para o vídeo usando a Wikipedia como fonte de pesquisa. Em seguida, o aplicativo gera o script com base no título e na pesquisa da Wikipedia. O título e o script gerados são exibidos ao usuário, juntamente com o histórico de títulos e scripts anteriores.

## Análise de Sentimento
Nesta opção, o aplicativo realiza a análise de sentimento de um determinado texto fornecido pelo usuário. O usuário insere o texto e o aplicativo utiliza um modelo pré-treinado para realizar a análise de sentimento. O resultado da análise, incluindo o sentimento identificado e a pontuação associada, é exibido ao usuário.

## Pergunte ao seu PDF
Nesta opção, o aplicativo permite ao usuário fazer perguntas sobre um arquivo PDF. O usuário carrega um arquivo PDF e o aplicativo extrai o texto do PDF. Em seguida, o aplicativo divide o texto em trechos menores e os utiliza como base de conhecimento para responder às perguntas do usuário. O aplicativo utiliza um modelo de perguntas e respostas para buscar respostas relevantes aos questionamentos feitos pelo usuário.

## Pergunte ao seu CSV
Nesta opção, o aplicativo permite ao usuário fazer perguntas sobre um arquivo CSV. O usuário carrega um arquivo CSV e o aplicativo exibe o conteúdo do arquivo em uma tabela interativa. O aplicativo também utiliza um modelo de linguagem para criar um agente capaz de responder a perguntas do usuário com base nos dados do arquivo CSV. O usuário pode inserir perguntas sobre o arquivo CSV e o aplicativo exibirá as respostas correspondentes.

## Como executar o aplicativo
Antes de executar o aplicativo, certifique-se de ter definido a variável de ambiente OPENAI_API_KEY com a chave de API da OpenAI. Para fazer isso, você pode adicionar o seguinte comando em seu terminal ou ambiente de desenvolvimento:

```bash
export OPENAI_API_KEY=sua_chave_de_api
```
Substitua **sua_chave_de_api** pela chave de API real fornecida pela OpenAI.

Após definir a variável de ambiente, é necessário ter o Streamlit e todas as dependências listadas no código instalados. Após a instalação das dependências, execute o seguinte comando:

```bash
streamlit run app.py
```
Isso iniciará o aplicativo e abrirá uma interface no navegador. A partir daí, o usuário pode selecionar uma das opções disponíveis e interagir com o aplicativo.
```
Created by Gustavo Oliveira
```