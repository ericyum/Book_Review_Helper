# Book_Review_Helper

 ---

  Phase 1: MVP(Minimum Viable Product) 구축 - 기본에 충실하기

  목표: 일단 작동하는 가장 간단한 버전의 RAG 챗봇을 만듭니다. 이것이 모든 개선의 기준점(Baseline)이 됩니다.


  액션:
   1. requirements.txt 파일에 langchain, openai, gradio, chromadb, tiktoken, pypdf 등 기본 라이브러리를 정의합니다.
   2. 사용자가 제공한 계획서의 기술 스택을 그대로 사용하여 app.py에 RAG 파이프라인을 구현합니다.
       * PyPDFLoader로 '어린 왕자' PDF 파일 로드
       * RecursiveCharacterTextSplitter로 텍스트 분할 (기본 chunk_size, chunk_overlap 값 사용)
       * OpenAIEmbeddings로 임베딩
       * Chroma 벡터 스토어에 저장
       * 기본적인 Prompt Template 사용
       * ChatOpenAI (e.g., gpt-3.5-turbo) 모델 사용
       * Gradio 인터페이스로 파일 업로드 및 채팅 기능 구현

  포트폴리오 스토리: "먼저, LangChain의 기본적인 구성요소들을 활용하여 RAG의 핵심 기능을 갖춘 프로토타입을 신속하게 구축했습니다."

  ---


  Phase 2: Chunking & Prompt 개선 - 가장 가성비 좋은 성능 향상

  목표: 적은 노력으로 답변의 질을 눈에 띄게 향상시킵니다. RAG에서 가장 중요한 '좋은 Context'를 LLM에 전달하는 부분에 집중합니다.


  액션:
   1. (Chunking 전략 변경):
       * 가설: "기본 청크 크기는 의미 단위로 잘리지 않아 답변의 정확도를 떨어뜨릴 것이다. 청크 크기와 중복 영역을 조절하면 문맥을 더 잘 파악할 수 있을 것이다."
       * 실험: chunk_size와 chunk_overlap 값을 바꿔가며 몇 가지 질문에 대한 답변의 질과 함께 제공되는 '근거(source)'가 어떻게 변하는지 비교하고 기록합니다. (e.g., (1000, 200) vs (500, 100))
   2. (Prompt Template 수정):
       * 가설: "LLM에 더 명확한 역할을 부여하고, Context 활용 방법을 구체적으로 지시하면 더 정확하고 일관된 답변을 생성할 것이다."
       * 실험:
           * Before: "Answer the question based on the context: {context} \n Question: {question}"
           * After: "You are a helpful assistant for writing a book report. Based on the provided context from the book, answer the user's question in Korean. If the answer is not in the context, say that you don't know. \n\n Context: {context} \n\n
             Question: {question}"
       * 두 프롬프트의 답변 차이를 비교 분석합니다.


  포트폴리오 스토리: "Baseline 모델의 답변 품질을 개선하기 위해, 가장 영향력이 큰 두 요소인 Chunking과 Prompt를 수정했습니다. Chunking 전략을 조절하여 검색 정확도를 높였고, Prompt Engineering을 통해 LLM이 주어진 문맥 내에서 충실하게 답변하도록 유도하여
   'Hallucination(환각)' 현상을 억제했습니다."

  ---

  Phase 3: Retriever 및 LLM 모델 변경 - 핵심 부품 교체

  목표: RAG 파이프라인의 핵심 부품을 교체하며 성능, 비용, 속도의 트레이드오프를 분석합니다.


  액션:
   1. (Retriever 교체):
       * 가설: "단순 Vector Search는 다양한 관점의 질문에 취약할 수 있다. MultiQueryRetriever를 사용하면 여러 관점에서 질문을 생성하여 더 풍부한 검색 결과를 얻을 수 있을 것이다."
       * 실험: 기본 VectorStoreRetriever와 MultiQueryRetriever의 검색 결과를 비교합니다. 복잡하고 미묘한 질문(e.g., "어린 왕자가 장미에게 느끼는 복합적인 감정은?")에 대해 어떤 Retriever가 더 적절한 문맥을 찾아오는지 평가합니다.
   2. (LLM 모델 변경):
       * 가설: "더 성능이 좋은 LLM 모델(e.g., gpt-4)은 검색된 문맥을 더 깊이 있게 이해하고 추론하여 우수한 품질의 답변을 생성할 것이다."
       * 실험: gpt-3.5-turbo와 gpt-4의 답변을 정성적으로 비교합니다. (비용과 응답 속도 변화도 함께 기록)


  포트폴리오 스토리: "답변 생성 능력 자체를 끌어올리기 위해 RAG의 핵심 엔진인 Retriever와 LLM을 교체하는 실험을 진행했습니다. Multi-Query Retriever를 도입하여 검색 성능을 향상시켰고, GPT-4 모델을 적용하여 최종 답변의 논리력과 표현력을 극대화했습니다.
  이 과정에서 성능, 비용, 속도 간의 균형점을 어떻게 찾아야 하는지에 대한 경험을 얻었습니다."

  ---

  Phase 4: 대화 기능 및 고급 RAG 적용 - 시스템 고도화


  목표: 단순 질의응답을 넘어, 대화의 흐름을 기억하고 더 발전된 RAG 방법론을 적용하여 프로젝트의 깊이를 더합니다.


  액션:
   1. (대화 메모리 추가):
       * 가설: "이전 대화 내용을 기억하지 못하면 사용자가 불편을 겪는다. 대화 메모리를 추가하면 더 자연스러운 사용자 경험을 제공할 수 있다."
       * 실험: ConversationBufferMemory를 RAG 체인에 추가하여, 후속 질문이 이전 질문의 맥락을 이해하고 답변하는지 확인합니다. (e.g., "그 등장인물에 대해 더 자세히 알려줘.")
   2. (고급 RAG 방법론 적용 - RAPTOR):
       * 가설: "RAPTOR 방법론을 적용하면, 개별 청크 수준을 넘어 책 전체의 주제나 플롯과 같은 추상적이고 종합적인 질문에도 효과적으로 답변할 수 있을 것이다."
       * 실험: RAPTOR 로직(텍스트를 재귀적으로 클러스터링하고 요약하여 트리 구조 생성)을 구현하고, "이 책의 전체적인 주제는 무엇이야?"와 같은 상위 레벨의 질문에 대한 답변이 어떻게 달라지는지 Phase 3의 결과와 비교합니다.


  포트폴리오 스토리: "마지막으로, 사용자 편의성을 높이기 위해 대화 메모리 기능을 추가하여 연속적인 질문이 가능하도록 했습니다. 더 나아가, 최신 RAG 연구인 RAPTOR 방법론을 적용하여, 세부 내용 검색을 넘어 책 전체를 아우르는 추상적인 질문까지 처리할 수
  있는 고도화된 시스템을 구축하며 프로젝트를 마무리했습니다."