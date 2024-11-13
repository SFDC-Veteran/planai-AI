import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const basicAcademicSearchRetrieverPrompt = `
아래 대화 내용과 후속 질문이 주어집니다. 후속 질문을 독립적인 질문으로 재구성하여, AI 모델이 학술 논문이나 기사를 검색할 수 있도록 해야 합니다. 다양한 학문적 주제에 대해 검색할 수 있도록 해야 하며, 단순한 글쓰기 작업이나 인사와 같은 질문에 대해서는 \`not_needed\`를 응답으로 반환해야 합니다.

예시:
1. 후속 질문: 안정적 확산은 어떻게 작동하나요?
재구성: 안정적 확산 작동 원리

2. 후속 질문: 선형대수학이 무엇인가요?
재구성: 선형대수학에 대한 개념

3. 후속 질문: 최신 심리학 연구 동향에 대한 자료를 제공해 주세요.
재구성: 최신 심리학 연구 동향에 관한 학술 자료

4. 후속 질문: 기후 변화에 대한 경제적 분석을 찾아 주세요.
재구성: 기후 변화에 대한 경제적 분석 논문

5. 후속 질문: AI의 윤리적 문제에 대해 논의한 자료가 있나요?
재구성: AI 윤리 문제에 대한 학술 논문

6. 후속 질문: 도시 계획에서 지속 가능성의 역할은 무엇인가요?
재구성: 도시 계획에서 지속 가능성의 역할에 관한 연구

7. 후속 질문: 최근의 교육 혁신에 관한 연구가 있나요?
재구성: 최근 교육 혁신에 관한 학술 연구

대화:
{chat_history}

후속 질문: {query}
재구성된 질문:
`;

const basicAcademicSearchResponsePrompt = `
    당신은 Perplexica라는 AI 모델로, 학술 논문 및 자료를 검색하여 사용자의 질문에 답변하는 전문가입니다. ‘학술 모드’로 설정되어 있어, 주로 다양한 분야의 논문과 학술 자료를 검색하는 방식으로 질문에 응답합니다.

    사용자 질문에 대해 유익하고 정확한 답변을 제공하기 위해, 주어진 문맥(context)을 사용하여 학문적으로 정확하고 유익한 정보를 제공해야 합니다. 문맥에는 검색 결과로 나온 논문 및 자료의 간단한 요약이 포함되어 있습니다. 이 문맥을 활용하여 질문에 답변하세요.

    답변은 중립적이고 학문적인 톤으로 작성되며, 한국어로 제공됩니다. 필요한 경우 Markdown을 사용하여 정보를 형식화하고, 리스트 형식으로 정보를 정리합니다. 각 문장에 해당하는 출처를 [숫자] 형식으로 인용하고, 출처 링크는 해당 문장 뒤에 제공합니다.

    사용자의 질문에 대해 관련된 논문 링크를 반드시 포함시켜 답변을 작성해야 합니다.

    <context>
    {context}
    </context>

    현재 날짜는 ${new Date().toISOString()}입니다.
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicAcademicSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicAcademicSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'kr',
        engines: ['arxiv', 'google scholar', 'pubmed'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content,
            metadata: {
              title: result.title,
              url: result.url,
              ...(result.img_src && { img_src: result.img_src }),
            },
          }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicAcademicSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicAcademicSearchRetrieverChain =
    createBasicAcademicSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    if (optimizationMode === 'speed') {
      return docsWithContent.slice(0, 15);
    } else if (optimizationMode === 'balanced') {
      const [docEmbeddings, queryEmbedding] = await Promise.all([
        embeddings.embedDocuments(
          docsWithContent.map((doc) => doc.pageContent),
        ),
        embeddings.embedQuery(query),
      ]);

      const similarity = docEmbeddings.map((docEmbedding, i) => {
        const sim = computeSimilarity(queryEmbedding, docEmbedding);

        return {
          index: i,
          similarity: sim,
        };
      });

      const sortedDocs = similarity
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 15)
        .map((sim) => docsWithContent[sim.index]);

      return sortedDocs;
    }
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicAcademicSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicAcademicSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicAcademicSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicAcademicSearchAnsweringChain =
      createBasicAcademicSearchAnsweringChain(
        llm,
        embeddings,
        optimizationMode,
      );

    const stream = basicAcademicSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in academic search: ${err}`);
  }

  return emitter;
};

const handleAcademicSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicAcademicSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleAcademicSearch;
