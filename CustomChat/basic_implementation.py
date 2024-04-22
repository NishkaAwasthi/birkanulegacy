from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import docx # to read word files --> pip install python-docx

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS # metaAI to compare embeddings/vector comparisons
from langchain_community.llms import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------------
# To read pdf
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# To read word
def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# To read txt
def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text
# ---------------------------------------------------------------------------------

train_directory = 'train_files/'
text = read_documents_from_directory(train_directory)
# extracts ALL information from text files in training folder into text string
# so text is one large string of ALL of the pdfs
#& print(len(text)) = 475138

# split text into chunks
char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, 
                                      chunk_overlap=200, length_function=len)
# use this to convert it into embeddings
text_chunks = char_text_splitter.split_text(text)

# create embeddings
# embeddings = OpenAIEmbeddings()
# docsearch = FAISS.from_texts(text_chunks, embeddings)

llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")
# ---------------------------------------------------------------------------------

# Ask a question
#query  = "What is the significance of 42?"
query  = "Who are the main characters in Jungle Book"

docs = docsearch.similarity_search(query )
 
response = chain.run(input_documents=docs, question=query )
print(" ")
print(query)
print(response)
  
#If you want to keep track of your spending
with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=query ) 
    print(cb)


#? What's the difference between embedding and fine tuning?
# In summary, the embedding approach is used for simpler models like semantic 
# search or recommendation models, where we just need to fetch the text similar 
# to the user's query rather than generate a new text like in AI Chatbot. So in 
# complex NLP tasks, we go with fine-tuning a pre-trained model.

#? So can this answer deep, meaningful questions about the theme of the novel?
# Probably not, since the model tries to find direct matches in the question 
# and the book. It cannot generate NEW ideas or concepts or interpretations not 
# directly in the book.

#? Wait so how is this even AI? Isn't this just a glorified Cmd-F?
