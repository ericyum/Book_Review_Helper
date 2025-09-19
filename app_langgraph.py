

import gradio as gr
import os
import pathlib
from typing import List, Dict, Annotated, Sequence, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.tools.tavily import TavilySearch
from langchain import hub
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

# --- 환경 설정 ---
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# Gemini API 키 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 전역 변수 및 모델 초기화 ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# --- 상태 정의 ---
class GraphState(TypedDict):
    """
    그래프의 상태를 나타냅니다.

    Attributes:
        question: 사용자의 질문
        generation: LLM이 생성한 답변
        documents: 검색된 문서 목록
    """
    question: str
    generation: str
    documents: List[Document]

# --- 평가기(Grader) 정의 ---

# 문서 관련성 평가기
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성에 대한 이진 점수."""
    binary_score: str = Field(description="문서가 질문과 관련이 있으면 'yes', 그렇지 않으면 'no'")

# 답변 사실성(Groundedness) 평가기
class GradeHallucinations(BaseModel):
    """생성된 답변에 환각(hallucination)이 있는지에 대한 이진 점수."""
    binary_score: str = Field(description="답변이 사실에 근거하면 'yes', 그렇지 않으면 'no'")

# 답변 유용성(Helpfulness) 평가기
class GradeAnswer(BaseModel):
    """답변이 질문을 해결하는지에 대한 이진 점수."""
    binary_score: str = Field(description="답변이 질문을 해결하면 'yes', 그렇지 않으면 'no'")

def get_graders(llm_choice: str):
    """선택된 LLM에 따라 평가기들을 반환합니다."""
    if llm_choice == "GPT-4o":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    # 문서 관련성 평가기
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    # 답변 사실성 평가기
    structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by the set of facts."),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

    # 답변 유용성 평가기
    structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a grader assessing whether an answer is useful to resolve a user question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a user question."),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_answer_grader
    
    return retrieval_grader, hallucination_grader, answer_grader

# --- 도구 정의 ---
web_search_tool = TavilySearch(max_results=3)

# --- 그래프 노드 정의 ---

def retrieve(state, retriever):
    """문서 검색"""
    print("---문서 검색---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state, llm_choice):
    """답변 생성"""
    print("---답변 생성---")
    question = state["question"]
    documents = state["documents"]
    
    if llm_choice == "GPT-4o":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    rag_chain = hub.pull("rlm/rag-prompt") | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state, retrieval_grader):
    """문서 관련성 평가"""
    print("---문서 관련성 확인---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---문서 관련성 점수: YES---")
            filtered_docs.append(d)
        else:
            print("---문서 관련성 점수: NO---")
    return {"documents": filtered_docs, "question": question}

def transform_query(state, llm_choice):
    """질문 변환"""
    print("---질문 변환---")
    question = state["question"]
    documents = state["documents"]

    if llm_choice == "GPT-4o":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    re_write_prompt = PromptTemplate(
        template="Provide a better question to retrieve relevant documents from a vector database. \n\n Original question: {question}",
        input_variables=["question"],
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
    
    return {"documents": documents, "question": better_question}

def web_search(state):
    """웹 검색"""
    print("---웹 검색---")
    question = state["question"]
    documents = state.get("documents", [])
    
    web_results = web_search_tool.invoke({"query": question})
    documents.extend([Document(page_content=d["content"], metadata={"source": d["url"]}) for d in web_results])
    
    return {"documents": documents, "question": question}

# --- 조건부 엣지 함수 ---

def decide_to_generate(state):
    """답변 생성 여부 결정"""
    print("---답변 생성 여부 결정---")
    if not state["documents"]:
        print("---결정: 문서 없음, 질문 변환---")
        return "transform_query"
    else:
        print("---결정: 문서 있음, 답변 생성---")
        return "generate"

def grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader):
    """생성된 답변의 사실성 및 유용성 평가"""
    print("---생성된 답변 평가---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---결정: 답변이 사실에 근거함---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---결정: 답변이 유용함---")
            return "useful"
        else:
            print("---결정: 답변이 유용하지 않음---")
            return "not useful"
    else:
        print("---결정: 답변이 사실에 근거하지 않음---")
        return "not supported"

# --- 그래프 생성 함수 ---

def create_base_rag_graph(llm_choice, retriever):
    """기본 RAG 그래프 생성"""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("generate", lambda state: generate(state, llm_choice))
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def create_crag_graph(llm_choice, retriever):
    """Corrective-RAG 그래프 생성"""
    retrieval_grader, _, _ = get_graders(llm_choice)
    
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, retrieval_grader))
    workflow.add_node("generate", lambda state: generate(state, llm_choice))
    workflow.add_node("transform_query", lambda state: transform_query(state, llm_choice))
    workflow.add_node("web_search", web_search)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def create_self_rag_graph(llm_choice, retriever):
    """Self-RAG 그래프 생성"""
    retrieval_grader, hallucination_grader, answer_grader = get_graders(llm_choice)

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, retrieval_grader))
    workflow.add_node("generate", lambda state: generate(state, llm_choice))
    workflow.add_node("transform_query", lambda state: transform_query(state, llm_choice))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        lambda state: grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    return workflow.compile()

# --- Gradio UI 및 로직 ---

def process_document(file, chunk_size, chunk_overlap):
    """PDF 처리 및 벡터스토어 생성"""
    if file is None:
        return None, "PDF 파일을 먼저 업로드해주세요.", gr.update(interactive=True)

    print(f"'{file.name}' 파일 처리 중...")
    file_path = pathlib.Path(file.name)
    
    try:
        # 최신 SDK는 genai.configure()를 사용하거나 환경 변수에서 API 키를 자동으로 읽습니다.
        # genai.Client()는 더 이상 사용되지 않는 패턴일 수 있습니다.
        uploaded_file = genai.upload_file(path=file_path, display_name=file_path.name)
        
        prompt_text = "pdf파일에서 텍스트 내용을 추출해줘. 왼쪽 페이지를 먼저 읽고, 오른쪽 페이지를 읽어와줘. 페이지 번호나 관련 없는 메타 정보는 제거하고 문장을 연결해줘."
        
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        response = model.generate_content([prompt_text, uploaded_file])
        extracted_text = response.text
    except Exception as e:
        return None, f"파일 처리 중 오류 발생: {e}", gr.update(interactive=True)

    if not extracted_text:
        return None, "PDF에서 텍스트를 추출하지 못했습니다.", gr.update(interactive=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    docs = text_splitter.create_documents([extracted_text])
    
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("파일 처리 완료.")
    return retriever, "파일 분석이 완료되었습니다. 이제 질문해주세요.", gr.update(interactive=False)

def chat_with_doc(message, history, retriever, llm_choice, rag_type):
    """LangGraph RAG 실행"""
    if retriever is None:
        history.append((message, "먼저 PDF 파일을 업로드하고 '분석 시작' 버튼을 눌러주세요."))
        return history, ""

    # 선택된 RAG 유형에 따라 그래프 생성
    if rag_type == "기본 RAG":
        app = create_base_rag_graph(llm_choice, retriever)
    elif rag_type == "Corrective RAG (CRAG)":
        app = create_crag_graph(llm_choice, retriever)
    elif rag_type == "Self RAG":
        app = create_self_rag_graph(llm_choice, retriever)
    else:
        history.append((message, "알 수 없는 RAG 유형입니다."))
        return history, ""

    inputs = {"question": message}
    final_state = app.invoke(inputs)
    
    answer = final_state.get("generation", "답변을 생성하지 못했습니다.")
    
    # 출처 문서 포맷팅
    source_text = ""
    if final_state.get("documents"):
        source_text += "--- 답변 근거 (출처) ---\n\n"
        for i, doc in enumerate(final_state["documents"]):
            source = doc.metadata.get('source', 'N/A')
            source_text += f"> **출처 {i+1}:** {source}\n\n"
            source_text += doc.page_content.strip()
            source_text += "\n\n---\n\n"

    history.append((message, answer))
    return history, source_text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    retriever_state = gr.State(None)

    gr.Markdown("## LangGraph 기반 RAG 독후감 보조 프로그램")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            
            gr.Markdown("### 설정")
            chunk_size_input = gr.Number(label="청크 크기", value=1000, interactive=True)
            chunk_overlap_input = gr.Number(label="청크 오버랩", value=100, interactive=True)
            
            llm_choice = gr.Radio(
                ["Gemini 1.5 Pro", "GPT-4o"],
                label="LLM 선택",
                value="Gemini 1.5 Pro",
                interactive=True
            )
            
            rag_type_selector = gr.Dropdown(
                ["기본 RAG", "Corrective RAG (CRAG)", "Self RAG"],
                label="RAG 아키텍처 선택",
                value="Corrective RAG (CRAG)",
                interactive=True
            )
            
            process_btn = gr.Button("분석 시작", variant="primary")
        
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="챗봇", height=600)
            with gr.Accordion("답변 근거 확인하기", open=False):
                source_display = gr.Markdown(label="출처 원문")
            with gr.Row():
                textbox = gr.Textbox(label="질문 입력", placeholder="책 내용에 대해 질문해보세요...", scale=7)
                submit_btn = gr.Button("질문하기", variant="primary", scale=1)
            clear_btn = gr.ClearButton([textbox, chatbot, source_display])

    process_btn.click(
        fn=process_document,
        inputs=[file_upload, chunk_size_input, chunk_overlap_input],
        outputs=[retriever_state, textbox, process_btn]
    )

    submit_btn.click(
        fn=chat_with_doc,
        inputs=[textbox, chatbot, retriever_state, llm_choice, rag_type_selector],
        outputs=[chatbot, source_display]
    ).then(lambda: "", outputs=[textbox])

    textbox.submit(
        fn=chat_with_doc,
        inputs=[textbox, chatbot, retriever_state, llm_choice, rag_type_selector],
        outputs=[chatbot, source_display]
    ).then(lambda: "", outputs=[textbox])

if __name__ == "__main__":
    demo.launch()
