#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import time
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def generate(query,  hide_source=False):
    # Parse the command line arguments
    print("Target source chunks: "+ str(target_source_chunks))

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    print("Retriever: "+ str(retriever))
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [MyCustomHandler()]
    print("Callbacks: "+ str(callbacks))
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True, streaming=True)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    print("LLM: "+ str(llm))
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not hide_source)  
    print("QA: "+ str(qa))
    # Start the generation process
    start = time.time()
    print("Start: "+ str(start))
    res = qa(query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']
    end = time.time()
    print("End: "+ str(end))

    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)
    return answer


def generate_data(query, hide_source=False):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # Create an instance of your custom handler
    custom_handler = MyCustomHandler()
    
    callbacks = [custom_handler]
    
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True, streaming=True)
        case _default:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not hide_source)
    
    # Start the generation process
    for chunk in generate_streamed_data(query, qa):
        yield chunk

def generate_streamed_data(query, qa):
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    yield f"\n\n> Question:\n{query}\n"
    yield f"\n> Answer (took {round(end - start, 2)} s.):\n{answer}\n"

    for document in docs:
        yield f"\n> {document.metadata['source']}:\n{document.page_content}\n"

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")
        # time.sleep(0.1)
        yield f"{token}"