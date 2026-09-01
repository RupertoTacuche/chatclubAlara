import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

with open("alara.txt", "r", encoding="utf-8") as f:
    documento = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(documento)

vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Retorna los 4 fragmentos mas relevantes
)


prompt = ChatPromptTemplate.from_template("""IA del Club Toastmasters Alara . Tu trabajo es responder preguntas
de los clientes UNICAMENTE usando la informacion proporcionada en el contexto.

Reglas estrictas:
1. SOLO responde con informacionn que estaen el contexto.
2. Si la pregunta no se puede responder con el contexto, di:
   "Lo siento, no tengo esa informaciÃ³n. Te recomiendo contactarnos
   por WhatsApp al 5526537776 con Everardo Martinez Perez"
3. Sé amable, conciso y Ãºtil.
4. Si preguntan precios, siempre menciona el precio exacto.
5. Responde en español.

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def respond(message, history):
    response = rag_chain.invoke(message)
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="Chat Inteligente del Club ToastMasters ALARA ",
    description="PregÃºntame sobre horarios, ubicaciÃ³n, eventos y mÃ¡s.",
    examples=[
        "¿Cuál es el dia y horario en que sesiona el club Alara?",
        "¿Qué es Toastmasters?",
        "¿Cuál es el costo para ingresar al club Alara?",
        "¿Cuál es el método usado para enseñar a hablar en público?",
        "¿Cómo se fundó el movimiento ToastMasters?",
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Documento cargado: {len(chunks)} fragmentos indexados")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme="soft"
    )
