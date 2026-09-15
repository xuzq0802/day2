import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import math

load_dotenv()

st.title("Exercise 2.2")

# 1. Allows the user to copy and paste two different texts.

if "text_1" not in st.session_state:
    st.session_state.text_1 = ""

if "text_2" not in st.session_state:
    st.session_state.text_2 = ""

# 新增：初始化两个 embedding 的存储位置
if "embedding_1" not in st.session_state:
    st.session_state.embedding_1 = None

if "embedding_2" not in st.session_state:
    st.session_state.embedding_2 = None

st.session_state.text_1 = st.text_area(
    label="Text 1"
)

# st.write(st.session_state.text_1)

st.session_state.text_2 = st.text_area(
    label="Text 2"
)

# st.write(st.session_state.text_2)

# 2. Creates an embedding for each chunk of text.

client = OpenAI()

# 修改：把返回结果存入 session_state
st.session_state.embedding_1 = client.embeddings.create(
    input=st.session_state.text_1,
    model="text-embedding-3-large"
)

#st.write(st.session_state.embedding_1.data[0].embedding)

# 修改：把返回结果存入 session_state
st.session_state.embedding_2 = client.embeddings.create(
    input=st.session_state.text_2,
    model="text-embedding-3-large"
)

#st.write(st.session_state.embedding_2.data[0].embedding)

# 3. Displays the cosine similarity of the two embeddings.

vector_1 = st.session_state.embedding_1.data[0].embedding
vector_2 = st.session_state.embedding_2.data[0].embedding

# 两个向量对应位置相乘，再求和
dot_product = sum(a * b for a, b in zip(vector_1, vector_2))

# 分别计算两个向量的长度
norm_1 = math.sqrt(sum(a * a for a in vector_1))
norm_2 = math.sqrt(sum(b * b for b in vector_2))

if norm_1 > 0 and norm_2 > 0:
    cosine_similarity = dot_product / (norm_1 * norm_2)
    st.write(f"Cosine similarity: {cosine_similarity:.6f}")
else:
    st.warning("Cannot calculate similarity because a vector has zero length.")