import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
from schematics.types import StringType
from supabase import create_client, Client
from bson import ObjectId
from schematics.models import Model
from database import db

url: str = 'https://kozjxvhnfeznacutgbsd.supabase.co'
key: str = ''



os.environ['OPENAI_API_KEY'] = ''
embeddings = OpenAIEmbeddings()


app = FastAPI()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'cred.json'


storage_client = storage.Client()

bucket_name = 'nyay_bucket_files'

dest_folder = 'docs'


def download(fileid):
  supabase: Client = create_client(url, key)

  with open("docs/" + fileid, 'wb+') as f:
    res = supabase.storage.from_('gyaan').download(fileid)
    f.write(res)

  return fileid


def loadpdf(file):
    loader = PyMuPDFLoader("./docs/{}".format(file))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts


class Item(BaseModel):
    name: str
    fileid: str
    desc: str
    cover_link: str

class Query(BaseModel):
    query: str
    context: str

class ImageQuery(BaseModel):
    data: str
    ctxname: str


class BotData(Model):
    bot_id= ObjectId()
    bot_name = StringType(required=True)
    bot_desc = StringType(required=True)
    bot_fileid = StringType(required=True)
    cover_photo = StringType(required=True)

newbot = BotData()


def create_bot(name, desc, fileid, cover_link):
    newbot.bot_id = ObjectId()
    newbot.bot_name  = name
    newbot.bot_desc = desc
    newbot.bot_fileid = fileid
    newbot.cover_photo = cover_link
    return dict(newbot)



def bot_helper(bot) -> dict:
    return {
        "id": str(bot["_id"]),
        "name": bot["bot_name"],
        "fileid": bot["bot_fileid"],
        "desc": bot["bot_desc"],
        "cover_photo": bot["cover_photo"],
    }



@app.get("/", status_code=201)
async def root():
    return {"message": "Team Delhi-bots is High"}


@app.post("/bot", status_code=201)
async def bot(item: Item):
    file = download(item.fileid)
    texts = loadpdf(file)
    persist_directory = file
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embeddings,
                                     persist_directory=persist_directory,
                                     )
    vectordb.persist()


    data = create_bot(item.name, item.desc, item.fileid, item.cover_link)
    dict(data)

    db.botdata.insert_one(data)

    print(file)
    return {"message": "done"}



@app.post("/query", status_code=201)
async def query(ask: Query):
    q = ask.query
    ctx = ask.context
    vectordb = Chroma(persist_directory=ctx, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    ask = f"###Prompt {q}"
    try:
        llm_response = qa(ask)
        print(llm_response["result"])
        return {"message": llm_response["result"]}
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
        return {"message": "something went wrong"}




@app.get("/getbot", status_code=201)
async def getbot():
    try:
        mdata = db["botdata"]
        data = mdata.find()
        final_data = []
        for x in data:
            tdata = bot_helper(x)
            final_data.append(tdata)
        return {"bots": final_data}
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
        return {"message": "something went wrong"}



@app.post("/imagebot", status_code=201)
async def getimagebot(imageData: ImageQuery):
    try:
        text = imageData.data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
        texts = text_splitter.split_text(text)
        doctexts = text_splitter.create_documents(texts)
        persist_directory = imageData.ctxname
        vectordb = Chroma.from_documents(documents=doctexts,
                                         embedding=embeddings,
                                         persist_directory=persist_directory,
                                         )
        vectordb.persist()
        return {"message": "SuccessFull done!"}
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
        return {"message": "something went wrong"}
