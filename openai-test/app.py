from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
import gradio as gr
import os
import json

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    num_outputs = 512

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs)

    index.storage_context.persist(persist_dir="./storage")

    return index

def chatbot(input_text):
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-trained AI Chatbot"
)

index = construct_index("docs")
iface.launch(share=True)