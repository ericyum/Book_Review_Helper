import gradio as gr
import os
import pathlib
from typing import List, Union
from operator import itemgetter

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from google import genai

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


# Google Generative AI 임베딩 모델 전역 초기화
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def process_document(file, chunk_size, chunk_overlap, separators_str, k_value):
    """
    업로드된 PDF 파일을 처리하여 retriever와 문서 전체 내용을 반환합니다.
    """
    if file is None:
        return None, None, "PDF 파일을 먼저 업로드해주세요.", "", "", gr.update(interactive=True)

    print(f"'{file.name}' 파일 처리 중...")
    
    # PDF 파일 업로드 및 텍스트 추출
    file_path = pathlib.Path(file.name)
    client = genai.Client()
    sample_file = client.files.upload(file=file_path)

    prompt_text = (
        "pdf파일에서 텍스트 내용을 추출해줘. pdf는 페이지가 나란히 배치되어 있어. "
        "왼쪽 페이지를 먼저 읽고, 오른쪽 페이지를 읽어와줘. "
        "그리고 페이지 번호나 책의 본문과 관련 없는 메타 정보는 제거하고 페이지간의 문장을 연결해줘."
    )

    print("Requesting content extraction from Gemini...")
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=[sample_file, prompt_text]
    )
    extracted_text = response.text

    # extracted_text가 None인지 확인하는 로직 추가
    if not extracted_text:
        error_message = "PDF 파일에서 텍스트를 추출하지 못했습니다. 파일에 텍스트가 포함되어 있는지 확인해주세요."
        print(error_message)
        return None, None, error_message, "", "", gr.update(interactive=True)

    # 텍스트를 청크로 분할
    separators_list = [s.strip() for s in separators_str.split(',')]
    if " " in separators_list:
        separators_list = [s if s != " " else "" for s in separators_list]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=separators_list
    )
    text_chunks = text_splitter.split_text(extracted_text)
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": int(k_value)}) # k_value 사용
    print("파일 처리 완료.")
    
    first_page_content = docs[0].page_content if docs else "내용 없음"
    page_indicator = f"Chunk 1 / {len(docs)}"
    
    return retriever, docs, "파일 분석이 완료되었습니다. 이제 질문해주세요.", first_page_content, page_indicator, gr.update(interactive=False)

def chat_with_doc(message, history, retriever, llm_choice):
    """
    채팅 함수. 답변과 함께 전체 출처 원문을 반환합니다.
    """
    if retriever is None:
        history.append((message, "먼저 PDF 파일을 업로드하고 '분석 시작' 버튼을 눌러주세요."))
        return history, ""

    if llm_choice == "Gemini 2.5 Pro":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    elif llm_choice == "GPT-4o":
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    # [HyDE] 가설 답변(요약)을 생성하는 프롬프트/체인 추가
    hyde_prompt = ChatPromptTemplate.from_template(
        "아래 질문에 대해 문서가 있다고 가정하고, 질문에 대한 간결한 가설 답변을 한국어로 3~5문장으로 작성하세요.\n\n질문: {question}"
    )
    hyde_chain = (
        {"question": itemgetter("question")}
        | hyde_prompt
        | llm
        | StrOutputParser()
    )
    # [HyDE] 끝

    prompt = ChatPromptTemplate.from_template('''
    당신은 주어진 Context를 바탕으로 질문에 답하는 AI 어시스턴트입니다.
    Context에는 질문과 관련된 검색 내용이 들어 있습니다. 그 정보를 사용하여 간결하고 정확하게 답변해주세요.
    절대 Context에 없는 내용을 지어내지 마세요. 답변은 항상 한국어로 작성해주세요.
                                      
    Context: {context}
    Question: {question}''')
    
    # [HyDE] 기존: itemgetter("question") | retriever  →  변경: hyde_chain | retriever
    retrieval_chain = (
        {
            "context": hyde_chain | retriever,   # [HyDE] 가설 답변을 검색 질의로 사용
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(
            answer=(
                {
                    "context": lambda x: x["context"],
                    "question": lambda x: x["question"],
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        )
    )
    # [HyDE] 끝

    questions = [q.strip() for q in message.split('\n') if q.strip()]
    
    full_answer = ""
    full_source_text = ""

    for i, q in enumerate(questions):
        print(f"\n--- Processing Question {i+1}: {q} ---")
        
        response = retrieval_chain.invoke({"question": q})
        answer = response["answer"]
        context_docs = response["context"]

        print(f"Retrieved Context for Question {i+1}:\n{context_docs}")

        full_answer += f"**질문 {i+1}: {q}**\n{answer}\n\n"
        
        if context_docs:
            full_source_text += f"--- 질문 {i+1} 답변 근거 (출처 원문) ---\n\n"
            for j, doc in enumerate(context_docs):
                page_num = doc.metadata.get('page', 'N/A')
                full_source_text += f"> **출처 {j+1} (Page: {page_num})**\n\n"
                full_source_text += doc.page_content.strip()
                full_source_text += "\n\n---\n\n"
            full_source_text += "\n"

    history.append((message, full_answer))
    return history, full_source_text

def change_page(docs, current_page_num, direction):
    """
    E-Book 뷰어의 페이지를 변경합니다.
    """
    if docs is None:
        return "", "", gr.update(interactive=True)
    
    new_page_num = current_page_num + direction
    if 0 <= new_page_num < len(docs):
        page_content = docs[new_page_num].page_content
        page_indicator = f"Chunk {new_page_num + 1} / {len(docs)}"
        return page_content, new_page_num, page_indicator
    
    # 페이지 범위를 벗어나면 현재 페이지 유지
    return docs[current_page_num].page_content, current_page_num, f"Chunk {current_page_num + 1} / {len(docs)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # 상태 변수들
    retriever_state = gr.State(None)
    docs_state = gr.State(None)
    page_num_state = gr.State(0)

    gr.Markdown("## RAG 독후감 보조 프로그램")

    with gr.Tabs():
        with gr.TabItem("챗봇"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
                    
                    gr.Markdown("### 청킹 설정")
                    chunk_size_input = gr.Number(label="청크 크기 (chunk_size)", value=500, interactive=True)
                    chunk_overlap_input = gr.Number(label="청크 오버랩 (chunk_overlap)", value=100, interactive=True)
                    separators_input = gr.Textbox(label="구분자 (쉼표로 구분)", value="\\n\\n, \\n, ., !, ?, ;, ,,  , ", interactive=True)
                    
                    gr.Markdown("### LLM 및 검색 설정")
                    # LLM 선택
                    llm_choice = gr.Radio(
                        ["Gemini 2.5 Pro", "GPT-4o"], # LLM 모델명 변경
                        label="LLM 선택",
                        value="Gemini 2.5 Pro",
                        interactive=True
                    )
                    # k 값 입력 컴포넌트 추가
                    k_value_input = gr.Number(label="검색 문서 수 (k)", value=3, interactive=True)
                    
                    process_btn = gr.Button("분석 시작", variant="primary")
                
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="챗봇", height=500)
                    with gr.Accordion("답변 근거 확인하기", open=False):
                        source_display = gr.Markdown(label="출처 원문")
                    with gr.Row():
                        textbox = gr.Textbox(label="질문 입력", placeholder="책 내용에 대해 질문해보세요...", scale=7)
                        submit_btn = gr.Button("질문하기", variant="primary", scale=1)
                    clear_btn = gr.ClearButton([textbox, chatbot, source_display])

        with gr.TabItem("E-Book 뷰어"):
            with gr.Column():
                ebook_viewer = gr.Markdown(label="문서 내용")
                with gr.Row():
                    prev_btn = gr.Button("이전 페이지")
                    page_indicator_text = gr.Textbox(label="페이지", interactive=False, scale=1)
                    next_btn = gr.Button("다음 페이지")

    # 이벤트 처리
    process_btn.click(
        fn=process_document,
        inputs=[file_upload, chunk_size_input, chunk_overlap_input, separators_input, k_value_input], # k_value_input 추가
        outputs=[retriever_state, docs_state, textbox, ebook_viewer, page_indicator_text, process_btn]
    )

    submit_btn.click(
        fn=chat_with_doc,
        inputs=[textbox, chatbot, retriever_state, llm_choice],
        outputs=[chatbot, source_display]
    ).then(lambda: "", outputs=[textbox])

    textbox.submit(
        fn=chat_with_doc,
        inputs=[textbox, chatbot, retriever_state, llm_choice],
        outputs=[chatbot, source_display]
    ).then(lambda: "", outputs=[textbox])

    prev_btn.click(
        fn=lambda docs, page_num: change_page(docs, page_num, -1),
        inputs=[docs_state, page_num_state],
        outputs=[ebook_viewer, page_num_state, page_indicator_text]
    )

    next_btn.click(
        fn=lambda docs, page_num: change_page(docs, page_num, 1),
        inputs=[docs_state, page_num_state],
        outputs=[ebook_viewer, page_num_state, page_indicator_text]
    )

if __name__ == "__main__":
    demo.launch()
