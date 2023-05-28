import logging

import pandas as pd
import streamlit as st
import transformers
from langchain.agents import create_pandas_dataframe_agent
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import FAISS
from PIL import Image
from PyPDF2 import PdfReader
from st_aggrid import AgGrid, GridOptionsBuilder

image = Image.open('src/logo.jpg')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """
    Streamlit app for LLM testing
    """
    st.set_page_config(page_title='🔗 gstw7', page_icon='🦜')

    st.sidebar.header('LLMs UI Apps')
    st.sidebar.image(image)
    add_selectbox = st.sidebar.radio(
        'Escolha seu app:',
        ('Criador de script para Youtube', 'Análise de Sentimento',
         'Pergunte ao seu PDF', 'Pergunte ao seu CSV'))

    if add_selectbox == 'Criador de script para Youtube':

        wiki = WikipediaAPIWrapper()

        st.title('📹 Criador de script para Youtube')
        prompt = st.text_input(
            'Sobre qual tema quer que eu escreva um script para um vídeo?')

        title_template = PromptTemplate(
            input_variables=['topic'],
            template='Escreva um título para um novo vídeo no Youtube sobre {topic}')

        script_template = PromptTemplate(
            input_variables=['title', 'wikipedia_research'],
            template='Escreva um script de vídeo do Youtube baseado neste título: {title}. \
                Aproveite esta pesquisa da Wikipedia para escrever o script: {wikipedia_research}')

        title_memory = ConversationBufferMemory(
            input_key='topic', memory_key='chat_history')
        script_memory = ConversationBufferMemory(
            input_key='title', memory_key='chat_history')

        llm = OpenAI(temperature=0.9)
        title_chain = LLMChain(llm=llm, prompt=title_template,
                               verbose=True, output_key='title', memory=title_memory)
        script_chain = LLMChain(llm=llm, prompt=script_template,
                                verbose=True, output_key='script', memory=script_memory)

        if prompt:
            title = title_chain.run(prompt)
            wiki_research = wiki.run(prompt)
            script = script_chain.run(
                title=title, wikipedia_research=wiki_research)

            st.write(title)
            st.write(script)

            with st.expander('Title History'):
                st.info(title_memory.buffer)

            with st.expander('Script History'):
                st.info(script_memory.buffer)

            with st.expander('Wikipedia Research'):
                st.info(wiki_research)

    if add_selectbox == 'Análise de Sentimento':

        # Paper: https://arxiv.org/pdf/2104.12250.pdf
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        map_sentiment = {
            'positive': 'Positivo',
            'negative': 'Negativo',
            'neutral': 'Neutro'
        }
        st.title('❤️💔 Análise de Sentimento')
        sa_llm = transformers.pipeline(
            "sentiment-analysis", model=model_path, tokenizer=model_path)

        text_analysis = st.text_input('Escreva seu texto aqui')

        if text_analysis:

            response = sa_llm(text_analysis)[0]

            label = response['label']
            score = response['score']

            sentiment = map_sentiment.get(label, 'Desconhecido')

            col1, col2 = st.columns(2)
            with col1:
                st.write(f'Sentimento: {sentiment}')
            with col2:
                st.write(f'Score: {round(score*100, 1)}%')

    if add_selectbox == 'Pergunte ao seu PDF':

        st.title('🎨 Pergunte ao seu PDF')
        pdf = st.file_uploader("Carregue seu PDF", type="pdf")

        if pdf:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len)
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            user_question = st.text_input("Faça uma pergunta sobre o seu PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs,
                                     question=user_question)
                st.write(response)

    if add_selectbox == 'Pergunte ao seu CSV':

        st.title("📈 Pergunte ao seu CSV")
        csv_file = st.file_uploader(
            "Carregue seu arquivo CSV com separador `;`", type="csv", help="Arquivos com separador `;`")

        if csv_file:

            try:
                df_file = pd.read_csv(csv_file, sep=';')
            except ValueError as error_value:
                st.error(str(error_value))
            except Exception:
                st.error(
                    "Ocorreu um erro ao ler o arquivo. Verifique se o arquivo é válido.")
            else:
                grid_build = GridOptionsBuilder.from_dataframe(df_file)
                grid_build.configure_pagination(enabled=True)
                grid_options = grid_build.build()
                AgGrid(df_file, gridOptions=grid_options,
                       height=250, theme='streamlit')

                llm = OpenAI(temperature=0.9)
                agent = create_pandas_dataframe_agent(
                    llm=llm, df=df_file, verbose=True)

                if len(df_file) > 0:

                    user_question = st.text_input(
                        "Faça uma pergunta sobre o seu CSV")

                    if user_question != "":
                        with st.spinner(text="Aguarde..."):
                            st.write(agent.run(user_question))
                else:
                    st.error(
                        "Ocorreu um erro ao ler o arquivo. Verifique se o arquivo é válido.")


if __name__ == '__main__':
    main()
