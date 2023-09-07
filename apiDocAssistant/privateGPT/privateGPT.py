#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import time
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def generate(query, hide_source=False, model_type="GPT4All"):
    # Parse the command line arguments
    print("Target source chunks: "+ str(target_source_chunks))

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = []
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            print("LlamaCpp model path: "+ str(model_path))
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx,n_ctx=2048, n_batch=model_n_batch, callbacks=callbacks, verbose=True)
        case "GPT4All":
            print("GPT4All model path: "+ str(model_path))
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True, streaming=True)
        case "ChatOpenAI":
            print("ChatOpenAI model path: "+ str(model_path))
            model_name = "gpt-3.5-turbo"
            llm = ChatOpenAI(model_name=model_name, verbose=True)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not hide_source)  

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
    return answer, docs
