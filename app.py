import gradio as gr
import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# .env 파일에서 환경변수 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings("ignore")

def process_and_chat(file, question):
    if file is None or question == "":
        return "오류: PDF 파일과 질문을 모두 입력해주세요.", ""

    if not os.getenv("OPENAI_API_KEY"):
        return "오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.", ""

    try:
        file_path = file.name
        
        # 1. Load
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 3. Embedding & Store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(texts, embeddings)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Reranker가 더 많은 문서를 검토하도록 k값 상향

        # 4. LLM, Prompt, Retriever, Chain 설정
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # MultiQueryRetriever 설정
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )

        # FlashRank Reranker 설정
        reranker = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=multi_query_retriever
        )
        
        # 프롬프트 템플릿 정의
        prompt_template = """
당신은 독후감 작성을 돕는 전문가입니다.
제공되는 책의 내용(Context)을 바탕으로 질문에 답해주세요.
Context에 질문과 관련 없는 내용(예: 역자 후기, 해설)이 포함되어 있다면, 해당 내용은 무시하고 이야기 본문에만 집중해서 답변을 생성하세요.
만약 Context에서 답을 찾을 수 없다면, '알 수 없음'이라고 답하세요.
답변은 반드시 한국어로 작성해주세요.

Question: {question}
Context: {context}
Answer:
"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain_type_kwargs = {"prompt": PROMPT}

        # 최종 QA 체인
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever, # Reranker가 적용된 최종 검색기 사용
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        # 6. 실행
        result = qa_chain({"query": question})
        
        answer = result["result"]
        source_docs = "\n\n--- \n\n".join([doc.page_content for doc in result["source_documents"]])
        
        return answer, source_docs

    except Exception as e:
        return f"처리 중 오류가 발생했습니다: {e}", ""

# Gradio UI 설정
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG (Multi-Query + Reranker) 독후감 보조 시스템")
    gr.Markdown("책 PDF 파일을 업로드하고, 내용에 대해 질문하여 독후감 작성에 필요한 아이디어를 얻어보세요.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            question_input = gr.Textbox(label="책 내용에 대해 질문하기", placeholder="예: 어린 왕자가 사는 B612 행성은 어떤 곳인가요?")
            submit_button = gr.Button("질문 전송", variant="primary")
        
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="AI 답변", lines=15, interactive=False)
            source_output = gr.Textbox(label="답변 근거 (Source Documents)", lines=15, interactive=False)

    submit_button.click(
        fn=process_and_chat,
        inputs=[pdf_input, question_input],
        outputs=[answer_output, source_output]
    )

if __name__ == "__main__":
    demo.launch()