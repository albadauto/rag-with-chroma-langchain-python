from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
DIR_DB = "db"

prompt_template = """
Responda a pergunta do usuário: {pergunta}
com base nessas informações:
{base_conhecimento}
Se você não encontrar informações pro usuário, responda não sei te dizer"""

def ask_user():
    ask = input("Escreva sua pergunta: ")
    embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=DIR_DB,embedding_function=embedding)

    results = db.similarity_search_with_relevance_scores(ask, k=3)

    text_result = []
    for result in results:
        text = result[0].page_content
        text_result.append(text)
    base_conhecimento = "\n\n----\n\n".join(text_result)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": ask, "base_conhecimento": base_conhecimento})
    #print(prompt)

    model = ChatOpenAI(model="gpt-4o-mini")
    text_response = model.invoke(prompt).content
    print("Resposta da IA: ", text_response)

ask_user()
