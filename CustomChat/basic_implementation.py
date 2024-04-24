from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import docx # to read word files --> pip install python-docx

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(text_chunks, embeddings)

llm = Openai()
chain = load_qa_chain(llm, chain_type="stuff")
# ---------------------------------------------------------------------------------

# Ask a question
query  = "Who are the main characters in Percy Jackson?"

docs = docsearch.similarity_search(query )
 
response = chain.run(input_documents=docs, question=query )
print(" ")
print(query)
print(response)


#? What's the difference between embedding and fine tuning?
# In summary, the embedding approach is used for simpler models like semantic 
# search or recommendation models, where we just need to fetch the text similar 
# to the user's query rather than generate a new text like in AI Chatbot. So in 
# complex NLP tasks, we go with fine-tuning a pre-trained model.

#? So can this answer deep, meaningful questions about the theme of the novel?
# Probably not, since the model tries to find direct matches in the question 
# and the book. It cannot generate NEW ideas or concepts or interpretations not 
# directly in the book.

# If the text contains explicit mentions or discussions of the main theme, and 
# if the question-answering model is trained well, it should be able to provide 
# a meaningful response. However, the accuracy of the response depends on 
# various factors such as the quality of the text extraction, the 
# comprehensiveness of the training data, and the capabilities of the 
# question-answering model.

#? Wait so how is this even AI? Isn't this just a glorified Cmd-F?
# While on the surface it might seem similar to a traditional text search 
# function like Cmd-F, the approach here involves several layers of AI 
# technology working together to provide a more nuanced and context-aware 
# response.

# Text Embeddings: The text from the documents is transformed into numerical 
# representations (embeddings) using a deep learning model. These embeddings 
# capture semantic meaning and relationships between words and sentences, going 
# beyond simple keyword matching.

# Similarity Search: The embeddings are then used to perform similarity search, 
# which goes beyond basic string matching. It finds documents that are 
# semantically similar to the query, even if they don't contain the exact same 
# words.

# Question Answering Model: The retrieved documents are then passed to a 
# question-answering model. This model is trained to understand natural language 
# questions and extract relevant information from text documents to provide 
# meaningful answers.

# Feedback Mechanisms: The system may incorporate feedback mechanisms to improve 
# its performance over time. For example, the get_openai_callback() function 
# could be used to provide feedback on the quality of the responses, which can 
# be used to fine-tune the underlying models.

# So, while the initial step of searching for text within documents may resemble 
# a basic text search, the subsequent steps involve advanced AI techniques that 
# enable the system to understand context, semantics, and provide more nuanced 
# answers.

#? So then can this model answer what is the main moral of the story?
