import gradio as gr
import os
import warnings
import numpy as np
import pandas as pd
import umap
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture

# .env 파일에서 환경변수 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# --- RAPTOR 구현부 (노트북 코드 기반) ---

# 1. 클러스터링 함수
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters

# 2. 요약 함수
def embed_cluster_summarize_texts(
    texts: List[str], level: int, embeddings_model: OpenAIEmbeddings, llm: ChatOpenAI
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text_embeddings = embeddings_model.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)

    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels

    expanded_list = []
    for index, row in df.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )
    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df["cluster"].unique()

    summarization_template = """다음은 책의 일부 내용입니다. 이 내용의 핵심을 한두 문장으로 요약해주세요.

내용:
{context}

요약:"""
    summarization_prompt = PromptTemplate.from_template(summarization_template)
    summarization_chain = summarization_prompt | llm | StrOutputParser()

    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = "\n---\n".join(df_cluster["text"].tolist())
        summaries.append(summarization_chain.invoke({"context": formatted_txt}))

    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )
    return df, df_summary

# 3. 재귀적 처리 함수
def recursive_embed_cluster_summarize(
    texts: List[str], embeddings_model: OpenAIEmbeddings, llm: ChatOpenAI, level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    results = {}
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level, embeddings_model, llm)
    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, embeddings_model, llm, level + 1, n_levels
        )
        results.update(next_level_results)
    return results

# --- Gradio 앱 구현부 ---
def process_and_chat_raptor(file, question):
    if file is None or question == "":
        return "오류: PDF 파일과 질문을 모두 입력해주세요.", ""
    if not os.getenv("OPENAI_API_KEY"):
        return "오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.", ""

    try:
        # 모델 초기화
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # 1. Load
        loader = PyPDFLoader(file.name)
        documents = loader.load()

        # 2. Split (초기 Leaf 노드 생성)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        leaf_texts = text_splitter.split_documents(documents)
        leaf_texts_content = [doc.page_content for doc in leaf_texts]

        # 3. RAPTOR 트리 구축
        results = recursive_embed_cluster_summarize(leaf_texts_content, embeddings_model, llm, level=1, n_levels=3)

        # 4. 모든 텍스트(원본+요약)를 VectorStore에 저장
        all_texts = leaf_texts_content.copy()
        for level in sorted(results.keys()):
            summaries = results[level][1]["summaries"].tolist()
            all_texts.extend(summaries)
        
        vector_store = Chroma.from_texts(texts=all_texts, embedding=embeddings_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 5. 최종 QA 체인 실행
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

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        result = qa_chain({"query": question})
        answer = result["result"]
        source_docs = "\n\n--- \n\n".join([doc.page_content for doc in result["source_documents"]])
        
        return answer, source_docs

    except Exception as e:
        return f"처리 중 오류가 발생했습니다: {e}", ""

# Gradio UI 설정
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAPTOR RAG 독후감 보조 시스템")
    gr.Markdown("책 PDF 파일을 업로드하고, RAPTOR 방식으로 처리된 내용에 대해 질문해보세요.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF 파일 업로드", file_types=[".pdf"])
            question_input = gr.Textbox(label="책 내용에 대해 질문하기", placeholder="예: 어린 왕자에게 있어 가시 달린 꽃은 어떤 존재인가요?")
            submit_button = gr.Button("질문 전송", variant="primary")
        
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="AI 답변", lines=15, interactive=False)
            source_output = gr.Textbox(label="답변 근거 (Source Documents)", lines=15, interactive=False)

    submit_button.click(
        fn=process_and_chat_raptor,
        inputs=[pdf_input, question_input],
        outputs=[answer_output, source_output]
    )

if __name__ == "__main__":
    demo.launch()
