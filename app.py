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

prompt = ChatPromptTemplate.from_template("""Eres el chatbot del Club ToastMaster Alara. Tu trabajo es responder preguntas de los clientes ÚNICAMENTE usando la información proporcionada en el contexto.

Reglas estrictas:
1. SOLO responde con información que esté en el contexto.
2. Si la pregunta no se puede responder con el contexto, di:
    "Lo siento, no tengo esa información. Te recomiendo contactarnos 
    por WhatsApp al +5526537776 con el Presidente del Club Alara: Everardo Martinez Perez"
3. Sé amable, conciso y útil.
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

# Estilos CSS con la regla añadida específicamente para el título principal
custom_css = """
:root {
    color-scheme: dark;
}
body, html, .gradio-container {
    background-color: #0b2545 !important;
    color: #ffffff !important;
}

.gradio-container, div.wrap, div.contain, .app {
    background-color: #0b2545 !important;
}

/* Forzar que el título principal de la cabecera sea blanco */
h1, .gr-header h1, .prose h1, header h1 {
    color: #ffffff !important;
}

.examples button, button.sample, .gr-samples table td button {
    background-color: #134074 !important;
    border: 1px solid #8da9c4 !important;
    color: #ffffff !important;
}

.examples button span, button.sample span, .gr-samples table td button span {
    color: #ffffff !important;
}

h2, h3, p, span, label {
    color: #ffffff !important;
}

.chatbot {
    background-color: #0b2545 !important;
    border-color: #134074 !important;
}
.message.user {
    background-color: #134074 !important;
    color: #ffffff !important;
}
.message.bot {
    background-color: #8da9c4 !important;
    color: #0b2545 !important;
}
textarea, input {
    background-color: #134074 !important;
    color: #ffffff !important;
}
"""

with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=respond,
        title="Club ToastMasters Alara - Chat Virtual",
        description="Pregúntame sobre horarios, ubicación y más.",
        examples=[
            "¿Que día sesiona el club Alara?",
            "¿Cuanto cuesta la membresia?",
            "¿Quienes pueden pertenecer?",
            "¿Cual es la tecnica para enseñar?",
            "¿Que es y de que trata un club Toastmasters?",
        ]
    )

if __name__ == "__main__":
    print(f"Documento cargado: {len(chunks)} fragmentos indexados")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=custom_css,
        theme=gr.themes.Default()
    )