import streamlit as st
import transformers
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper


def main():
    """
    Streamlit app for LLM testing
    """
    st.set_page_config(page_title='🔗 INTNEG-GPT', page_icon='🦜')

    add_selectbox = st.sidebar.selectbox(
        'Qual modelo quer testar?',
        ('Youtube GPT Creator', 'Sentiment GPT Analysis', 'PDF GPT Creator', 'CSV GPT Creator'))

    if add_selectbox == 'Youtube GPT Creator':

        wiki = WikipediaAPIWrapper()

        st.title('📹 Youtube GPT Creator')
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

    if add_selectbox == 'Sentiment GPT Analysis':

        # Paper: https://arxiv.org/pdf/2104.12250.pdf
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        map_sentiment = {
            'positive': 'Positivo',
            'negative': 'Negativo',
            'neutral': 'Neutro'
        }
        st.title('❤️ Sentiment Analysis')
        sa_llm = transformers.pipeline(
            "sentiment-analysis", model=model_path, tokenizer=model_path)

        text_analysis = st.text_input('Escreva seu texto aqui', )

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


if __name__ == '__main__':
    main()
