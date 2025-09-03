import gradio as gr
import os
import uuid
import base64
import io
import warnings
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from PIL import Image

# .env 파일에서 환경변수 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# --- 1. PDF 요소 추출 및 분류 ---
def extract_pdf_elements(pdf_path, image_output_dir):
    return partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_output_dir,
    )

def categorize_elements(raw_pdf_elements):
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

# --- 2. 텍스트 및 이미지 요약 생성 ---
def generate_text_summaries(texts, tables, llm):
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()
    
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5}) if texts else []
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5}) if tables else []
    return text_summaries, table_summaries

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, llm_vision):
    msg = llm_vision.invoke(
        [HumanMessage(content=[
            {"type": "text", "text": "Describe the image in detail. Be specific about graphs, charts, and any text present."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"},
        ])]
    )
    return msg.content

# --- 3. Multi-Vector Retriever 생성 ---
def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    def add_documents(doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries: add_documents(text_summaries, texts)
    if table_summaries: add_documents(table_summaries, tables)
    if image_summaries: add_documents(image_summaries, images)
    return retriever

# --- 4. RAG 체인 및 Gradio 앱 ---
def resize_base64_image(base64_string, size=(1280, 720)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        doc_content = doc.page_content
        if doc_content.startswith("iVBORw0KGgo") or doc_content.startswith("/9j/"):
             b64_images.append(resize_base64_image(doc_content))
        else:
            texts.append(doc_content)
    return {"images": b64_images, "texts": texts}

def create_multimodal_rag_chain(retriever):
    llm_vision = ChatOpenAI(model="gpt-4o", max_tokens=2048)
    
    def img_prompt_func(data_dict):
        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages = []
        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"]:
                messages.append({"type": "image_url", "image_url": f"data:image/png;base64,{image}"})
        messages.append({"type": "text", "text": f"You are an expert book reviewer. Use the following text and images from a book to answer the question. Answer in Korean.\n\nUser Question: {data_dict['question']}\n\nText Context:\n{formatted_texts}"})
        return [HumanMessage(content=messages)]

    chain = (
        {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
        | RunnableLambda(img_prompt_func)
        | llm_vision
        | StrOutputParser()
    )
    return chain

def process_and_chat_multimodal(file, question):
    if file is None or question == "":
        return "오류: PDF 파일과 질문을 모두 입력해주세요.", "", [], None
    if not os.getenv("OPENAI_API_KEY"):
        return "오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.", "", [], None

    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []  # 오류 발생 시 참조할 수 있도록 변수를 미리 초기화합니다.
    try:
        # 모델 초기화
        llm_text = ChatOpenAI(model="gpt-4", temperature=0)
        llm_vision = ChatOpenAI(model="gpt-4o", max_tokens=2048)
        embeddings = OpenAIEmbeddings()

        # 1. PDF에서 요소 추출
        raw_pdf_elements = extract_pdf_elements(file.name, temp_dir)
        texts, tables = categorize_elements(raw_pdf_elements)

        # 2. 요약 생성
        text_summaries, table_summaries = generate_text_summaries(texts, tables, llm_text)
        
        image_paths = sorted([os.path.join(temp_dir, img) for img in os.listdir(temp_dir) if img.endswith(".jpg")])
        image_base64_list = [encode_image(p) for p in image_paths]
        image_summaries = [image_summarize(b64, llm_vision) for b64 in image_base64_list]

        # 3. Retriever 생성
        vectorstore = Chroma(collection_name="multimodal_rag", embedding_function=embeddings)
        retriever = create_multi_vector_retriever(
            vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, image_base64_list
        )

        # 4. RAG 체인 생성 및 실행
        chain = create_multimodal_rag_chain(retriever)
        answer = chain.invoke(question)
        
        # 5. 근거 자료 반환 (텍스트 및 이미지)
        retrieved_docs = retriever.invoke(question)
        retrieved_context = split_image_text_types(retrieved_docs)
        
        return answer, "\n---\n".join(retrieved_context["texts"]), retrieved_context["images"], None

    except Exception as e:
        return f"처리 중 오류가 발생했습니다: {e}", "", [], None
    finally:
        # 임시 이미지 파일 및 디렉토리 정리
        for path in image_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 멀티모달 RAG 독후감 보조 시스템")
    gr.Markdown("책 PDF를 업로드하면, 텍스트와 이미지를 모두 분석하여 질문에 답변합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            question_input = gr.Textbox(label="책 내용에 대해 질문하기", placeholder="예: 코끼리를 삼킨 보아뱀 그림에 대해 설명해줘.")
            submit_button = gr.Button("질문 전송", variant="primary")
        
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="AI 답변", lines=10, interactive=False)
            source_text_output = gr.Textbox(label="답변 근거 (텍스트)", lines=10, interactive=False)
            source_images_output = gr.Gallery(label="답변 근거 (이미지)", object_fit="contain", height="auto")
            error_output = gr.Textbox(label="오류", visible=True)

    submit_button.click(
        fn=process_and_chat_multimodal,
        inputs=[pdf_input, question_input],
        outputs=[answer_output, source_text_output, source_images_output, error_output]
    )

if __name__ == "__main__":
    demo.launch()