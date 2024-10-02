import logging
import inspect
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain

from .memory import CustomMemory
from .file_processing import process_file  # Import the file processing function

# ToDo: add Open API key 
openai_api_key = "putAPIkey here"

log = logging.getLogger(__name__)

def create_prompt() -> PromptTemplate:
    """Creates a prompt for the retrieval chain."""
    prompt_instructions = """
    Respond to the question as accurately as possible using only the context provided.
    
    Context:
    {context}
    
    Question: {input}
    
    Conversation:
    {chat_history}
    """

    template = inspect.cleandoc(prompt_instructions)
    input_variables = ["input", "context", "chat_history"]

    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables
    )
    return prompt

def search(question: str, document_url: str) -> dict[str, Any]:
    # Memory which saves/provides context of the conversation to the prompt
    log.info("Creating memory object.")
    memory = CustomMemory(memory_key="chat_history")

    # LLM to call.
    log.info("Creating an LLM to call.")
    llm = ChatOpenAI(
        openai_api_key="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",
        temperature=0,
        model="gpt-4"
    )

    # Prompt that will be used to call the LLM
    log.info("Creating prompt to pass to chain.")
    prompt = create_prompt()

    # Combines retrieved documents
    log.info("Creating chain that combines retrieved documents.")
    combine_docs_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",  # Modify as per your specific implementation
        document_variable_name="context",
        verbose=True,
        prompt=prompt,
    )

    # Processing the document using its URL
    log.info("Processing the document.")
    file_content = process_file(document_url)

    # Creating Search chain
    log.info("Creating Search chain.")
    rag = RetrievalQAWithSourcesChain(
        memory=memory,
        combine_documents_chain=combine_docs_chain,
        question_key="input",
        answer_key="output",
        sources_answer_key="sources",
        max_tokens_limit=3375,
        reduce_k_below_max_tokens=True,
        return_source_documents=True,
    )

    # Executing the search with combined input
    combined_input = f"Question: {question}\nFile Content: {file_content}"
    log.info("Executing the search.")
    response_dict = rag.invoke({"input": combined_input})
    log.info("Search executing successful.")

    return response_dict

# Rest of the code (if any) remains here
