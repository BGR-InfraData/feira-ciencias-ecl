import logging

import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI
from PIL import Image
from PyPDF2 import PdfReader

from utils.constants import PROMPT_QUESTION_AI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

image = Image.open('src/equipe.jpg')

st.set_page_config(page_title='ECL - Feira de Ciências', page_icon='🤖')
msgs = StreamlitChatMessageHistory(key="langchain_messages")


def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


def main():
    """
    Streamlit app for LLM testing
    """

    st.sidebar.header("INTELIGÊNCIA ARTIFICIAL")
    st.sidebar.subheader(
        "Considerações sobre o uso no cotidiano", divider=True)
    st.sidebar.text("""
                     Feira de Ciências 
                     Esola Conceição Lyra
                     
                     Professora Rejane Soares
                     
                     Alunas:
                     Ana Clara Anacleto, 
                     Anny Gabrielly, 
                     Gabriela Araujo, 
                     Isabelly Mendes, 
                     Maria Alice Oliveira, 
                     Maria Luiza Clemente, 
                     Sofia Peixoto
                     
                     6º ano""")
    st.sidebar.subheader("", divider=True)

    add_selectbox = st.sidebar.radio(
        'Apps com Inteligência Artificial (IA):',
        ('Questionário', 'Pergunte ao seu PDF', 'Criador de script para Youtube'))

    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.image(image)

    if add_selectbox == 'Criador de script para Youtube':

        st.title('📹 Criador de script para Youtube')
        prompt = st.text_input(
            'Sobre qual tema quer que eu escreva um script para um vídeo?')

        title_template = PromptTemplate(
            input_variables=['topic'],
            template='Você é um criador de conteúdo para vídeos no Youtube e deverá escrever um título e os tópicos que deverão ser abordados para um novo vídeo no Youtube sobre o seguinte tópico: {topic}')

        llm = OpenAI(temperature=0)
        title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

        if prompt:
            title = title_chain.run(prompt)
            script = title_chain.run(
                topic=title)

            st.write(title)
            st.write(script)

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

                with st.expander('History Ask'):
                    st.info(user_question)

                with st.expander('Result History'):
                    st.info(response)

    if add_selectbox == 'Questionário':

        st.title("🤖 Questionário sobre Inteligência Artificial")

        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        if len(msgs.messages) == 0:
            msgs.add_ai_message(
                "Oi, antes de começar me diz qual o seu o nome?")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_QUESTION_AI),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | ChatOpenAI(model='gpt-4o')
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="history",
        )

        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input():
            st.chat_message("human").write(prompt)
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": prompt}, config)
            st.chat_message("ai").write(response.content)

        with st.sidebar.expander('Histórico'):
            st.sidebar.button('Limpar histórico', on_click=clear_cache)


if __name__ == '__main__':
    main()
