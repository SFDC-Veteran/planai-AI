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
아래 대화 내용과 후속 질문이 주어집니다. 후속 질문이 독립적인 질문이 되도록 재구성하여, AI 모델이 학술 논문이나 기사를 검색할 수 있도록 해야 합니다.
일반적인 글쓰기 작업이나 간단한 인사(예: '안녕하세요', '반갑습니다')와 같은 질문이 아닐 경우, \`not_needed\`를 응답으로 반환해야 합니다.

예시:
1. 후속 질문: 안정적 확산은 어떻게 작동하나요?
재구성: 안정적 확산 작동 원리

2. 후속 질문: 선형대수학이 무엇인가요?
재구성: 선형대수학

3. 후속 질문: 최근 AI 발전에 대한 자료를 제공해 주세요.
재구성: 최근 AI 발전에 대한 학술 자료

4. 후속 질문: 강화학습에 관한 학술 논문을 찾아 주세요.
재구성: 강화학습에 관한 학술 논문

대화:
{chat_history}

후속 질문: {query}
재구성된 질문:
`;

const basicAcademicSearchResponsePrompt = `
    당신은 Perplexica라는 AI 모델이며, 논문 및 학술 아티클을 검색하여 사용자의 질문에 답변하는 전문가입니다. ‘학술 모드’로 설정되어 있어 주로 논문과 학술 자료를 중심으로 검색합니다.

    사용자 질문에 대한 답변은 주어진 문맥(context)을 사용하여 학문적으로 정확하고 유익하게 작성해야 합니다. 문맥에는 해당 페이지의 간단한 요약이 포함된 검색 결과가 있습니다.
    이 문맥을 활용하여 사용자 질문에 대해 유익한 답변을 작성하세요. 각 답변에는 관련된 학술 논문이나 자료의 링크를 반드시 포함하여, 사용자가 해당 출처를 참고할 수 있도록 해야 합니다. 답변은 중립적이고 학문적인 톤을 유지하며, 한국어로 작성하세요.

    Markdown을 사용해 정보를 형식화하고, 필요한 경우 정보는 리스트 형식으로 정리하세요. 각 문장에 해당하는 출처를 [숫자] 형식으로 인용하고, 출처 링크는 해당 문장 바로 뒤에 제공하여 출처를 명확히 합니다.
    
    사용자의 질문에 대해 유익한 논문 링크를 필수적으로 포함하여 답변을 완성하세요.

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
