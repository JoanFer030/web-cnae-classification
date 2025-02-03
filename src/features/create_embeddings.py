import os
import sys
import tiktoken
from tqdm import tqdm
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.files import open_web, list_webs

########################################################
###                  CREATE EMBEDDING                ###
########################################################
def chunk_text(text: str, max_tokens: int, model: str = "gpt-4"):
    """
    Divide the text in chunks of the maximum number of tokens.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    last = 0
    chunks = []
    while last <= len(tokens):
        n_last = last + max_tokens
        truncated_tokens = tokens[last:n_last]
        chunk = encoding.decode(truncated_tokens)
        chunks.append(chunk)
        last = n_last
    return chunks

def get_embedding(client: OpenAI, text: str, model: str = "text-embedding-3-small"):
    """
    Performs the GET request to the Open AI API and returns the embedding vector.
    """
    try:
        response = client.embeddings.create(
            input = text,
            model = model
        )
        return response.data[0].embedding
    except:
        return []

def save_embedding(embedding: list[float], nif: str, folder_path: str):
    """
    It recives a vector containing the embedding and saves it into a 
    plain text file where each line contains a value. The created 
    file is named with the nif of the company.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = folder_path + nif + ".txt"
    processed = "\n".join([str(e) for e in embedding])
    with open(save_path, "w") as file:
        file.write(processed)

def generate_embeddings(client: OpenAI, folder_path: str, save_path: str):
    """
    Lists all the stored webs, divide the text in chunks of the
    maximum number of tokens, calculates the embeddig for each
    chunk and stores all in one file.
    """
    webs = 0
    webs_list = list_webs(folder_path)
    for i in tqdm(range(len(webs_list)), desc = "Creating vector embeddings"):
        nif, web_path = webs_list[i]
        text = open_web(web_path)
        chunks = chunk_text(text, 8191)
        embedding = []
        for chunk in chunks:
            temp_vector = get_embedding(client, chunk)
            embedding.extend(temp_vector)
        if embedding:
            webs += 1
            save_embedding(embedding, nif, save_path)
    print(f"Created embedding vector for {webs:,} web sites in {save_path}")
    print(f"{len(webs_list) - webs:,} went wrong")

########################################################
###                  MAIN FUNCTIONS                  ###
########################################################
def create_embeddings(config: dict, secrets: dict):
    base_path = config["features"]["base_path"]
    processed_path = base_path + config["features"]["original"]
    embeddings_path = base_path + config["features"]["embeddings"]
    api_key = secrets["open_ai_api_key"]
    client = OpenAI(api_key = api_key)
    generate_embeddings(client, processed_path, embeddings_path)