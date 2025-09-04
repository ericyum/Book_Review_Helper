import gradio as gr
import os
import pathlib
from typing import List, Union

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI # Added this import

from google import genai

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')



# Google Generative AI 임베딩 모델 전역 초기화
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def process_document(file, chunk_size, chunk_overlap, separators_str):
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

    prompt = (
        "pdf파일에서 텍스트 내용을 추출해줘. pdf는 페이지가 나란히 배치되어 있어. "
        "왼쪽 페이지를 먼저 읽고, 오른쪽 페이지를 읽어와줘. "
        "그리고 페이지 번호나 책의 본문과 관련 없는 메타 정보는 제거하고 페이지간의 문장을 연결해줘."
    )

    print("Requesting content extraction from Gemini...")
    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=[sample_file, prompt]
    )
    extracted_text = response.text

    # 텍스트를 청크로 분할
    separators_list = [s.strip() for s in separators_str.split(',')]
    # Handle the empty string separator if it's explicitly provided
    # The user's default value for separators_input has " " for empty string.
    # So, if " " is in the list, replace it with ""
    if " " in separators_list:
        separators_list = [s if s != " " else "" for s in separators_list]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size), # Ensure it's an integer
        chunk_overlap=int(chunk_overlap), # Ensure it's an integer
        separators=separators_list
    )
    text_chunks = text_splitter.split_text(extracted_text)
    docs = [Document(page_content=chunk) for chunk in text_chunks] # Use langchain_core.documents.Document

    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15}) # Increased k to 15
    print("파일 처리 완료.")
    
    # E-Book 뷰어는 이제 원본 PDF 페이지가 아닌 텍스트 청크를 표시합니다.
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

    if llm_choice == "Gemini 1.5 Flash":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif llm_choice == "GPT-3.5 Turbo (ChatGPT)":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        # Default to Gemini 1.5 Flash if an unexpected choice is made
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template('''
    당신은 주어진 <context>를 바탕으로 질문에 답하는 AI 어시스턴트입니다.
    <context>에 질문에 대한 검색 내용을 넣어주세요. 그 정보를 사용하여 간결하고 정확하게 답변해주세요.
    절대 <context>에 없는 내용을 지어내지 마세요. 답변은 항상 한국어로 작성해주세요.
                                          
    <context>{context}</context>
    Question: {input}''')
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Handle multiple questions
    questions = [q.strip() for q in message.split('\n') if q.strip()]
    
    full_answer = ""
    full_source_text = ""

    for i, q in enumerate(questions):
        print(f"\n--- Processing Question {i+1}: {q} ---") # Debug print
        response = retrieval_chain.invoke({"input": q})
        answer = response["answer"]
        context_docs = response["context"]

        print(f"Retrieved Context for Question {i+1}:\n{context_docs}") # Debug print

        full_answer += f"**질문 {i+1}: {q}**\n{answer}\n\n"
        
        if context_docs:
            full_source_text += f"--- 질문 {i+1} 답변 근거 (출처 원문) ---\n\n"
            for j, doc in enumerate(context_docs):
                page_num = doc.metadata.get('page', 'N/A')
                full_source_text += f"> **출처 {j+1} (Page: {page_num})**\n\n"
                full_source_text += doc.page_content.strip()
                full_source_text += "\n\n---\n\n"
            full_source_text += "\n" # Add a newline between sources for different questions

    history.append((message, full_answer))
    return history, full_source_text

def change_page(docs, current_page_num, direction):
    """
    E-Book 뷰어의 페이지를 변경합니다.
    """
    if docs is None:
        return "", ""
    
    new_page_num = current_page_num + direction
    if 0 <= new_page_num < len(docs):
        page_content = docs[new_page_num].page_content
        page_indicator = f"Page {new_page_num + 1} / {len(docs)}"
        return page_content, new_page_num, page_indicator
    
    # 페이지 범위를 벗어나면 현재 페이지 유지
    return docs[current_page_num].page_content, current_page_num, f"Page {current_page_num + 1} / {len(docs)}"

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
                    chunk_size_input = gr.Number(label="청크 크기 (chunk_size)", value=250, interactive=True)
                    chunk_overlap_input = gr.Number(label="청크 오버랩 (chunk_overlap)", value=50, interactive=True)
                    separators_input = gr.Textbox(label="구분자 (쉼표로 구분)", value="\\n\\n, \\n, ., !, ?, ;, ,,  , ", interactive=True)
                    
                    gr.Markdown("### LLM 선택")
                    llm_choice = gr.Radio(
                        ["Gemini 1.5 Flash(일반)", "GPT-3.5 Turbo (추론)"],
                        label="LLM 선택",
                        value="Gemini 1.5 Flash",
                        interactive=True
                    )
                    
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
        inputs=[file_upload, chunk_size_input, chunk_overlap_input, separators_input],
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
