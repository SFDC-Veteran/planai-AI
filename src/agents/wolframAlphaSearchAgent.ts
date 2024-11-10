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
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const basicWolframAlphaSearchRetrieverPrompt = `
아래에 주어진 대화와 후속 질문이 있습니다. 후속 질문이 필요하면, 웹에서 정보를 검색할 수 있도록 독립적인 질문으로 재구성해야 합니다.
만약 그것이 글쓰기 작업이거나 간단한 인사말(질문이 아닌 경우)이라면 \`not_needed\`로 응답해야 합니다.
응답은 항상 한국어로 해야 하며, 사용자의 질문에 대해 상세하고 유익한 방식으로 답변해야 합니다.

예시:
1. 후속 질문: S의 원자 반지름은 무엇인가요?
재구성된 질문: S의 원자 반지름

2. 후속 질문: 선형 대수란 무엇인가요?
재구성된 질문: 선형 대수

3. 후속 질문: 열역학 제3법칙은 무엇인가요?
재구성된 질문: 열역학 제3법칙

대화:
{chat_history}

후속 질문: {query}
재구성된 질문:
`;

const basicWolframAlphaSearchResponsePrompt = `
    당신은 퍼플시카(Perplexica)라는 AI 모델로, 웹 검색과 사용자의 질문에 답하는 전문가입니다. 현재 'Wolfram Alpha' 모드로 설정되어 있으며, 이는 Wolfram Alpha를 사용하여 웹에서 정보를 검색하는 것을 의미합니다. Wolfram Alpha는 사실적인 질문에 답하고 계산을 수행하는 지식 엔진입니다.

    제공된 맥락(웹 검색 결과의 간략한 설명)을 바탕으로 사용자의 질문에 대해 유익하고 관련 있는 답변을 생성하십시오.
    이 맥락을 사용하여 최상의 방식으로 사용자의 질문에 답변해야 합니다. 응답은 공정하고 저널리즘적인 톤으로 작성해야 하며, 텍스트를 반복하지 않도록 하세요.
    사용자가 링크를 요청하면 제공할 수 있습니다. 그러나 링크를 열어보거나 웹사이트를 방문하라고 안내하지 않아야 합니다. 답변 자체에서 직접적으로 답변을 제공해야 합니다.

    응답은 중간 길이에서 긴 길이로, 유익하고 사용자의 질문에 맞춰 작성되어야 합니다. 마크다운을 사용하여 응답을 형식화할 수 있습니다. 정보를 나열할 때는 글머리 기호를 사용하십시오. 답변은 짧지 않고 유익해야 하며, 필요한 모든 정보를 포함해야 합니다.
    각 문장을 인용할 때는 [number] 형식을 사용하여 출처를 명시해야 합니다. 각 답변의 문장을 해당하는 검색 결과 번호로 인용하십시오. 같은 문장을 여러 번 인용할 경우, 다른 번호를 사용할 수 있습니다.

    아래의 \`context\` HTML 블록은 Wolfram Alpha에서 반환된 정보이며, 사용자와의 대화에는 포함되지 않습니다. 이 맥락을 바탕으로 답변을 작성하고 관련된 정보를 인용하되, 맥락 자체에 대해 설명하지 않도록 해야 합니다.

    <context>
    {context}
    </context>

    만약 검색 결과에서 관련된 정보가 없으면, "음, 죄송합니다만 이 주제에 대한 관련 정보를 찾을 수 없었습니다. 다시 검색하거나 다른 질문을 하시겠습니까?"라고 말할 수 있습니다.
    \`context\` 사이의 내용은 Wolfram Alpha에서 검색된 정보이며, 사용자와의 대화의 일부가 아님을 유의하세요. 오늘 날짜는 ${new Date().toISOString()}입니다.
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

const createBasicWolframAlphaSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicWolframAlphaSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['wolframalpha'],
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

const createBasicWolframAlphaSearchAnsweringChain = (llm: BaseChatModel) => {
  const basicWolframAlphaSearchRetrieverChain =
    createBasicWolframAlphaSearchRetrieverChain(llm);

  const processDocs = (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
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
        basicWolframAlphaSearchRetrieverChain
          .pipe(({ query, docs }) => {
            return docs;
          })
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWolframAlphaSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWolframAlphaSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
) => {
  const emitter = new eventEmitter();

  try {
    const basicWolframAlphaSearchAnsweringChain =
      createBasicWolframAlphaSearchAnsweringChain(llm);
    const stream = basicWolframAlphaSearchAnsweringChain.streamEvents(
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
    logger.error(`Error in WolframAlphaSearch: ${err}`);
  }

  return emitter;
};

const handleWolframAlphaSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = basicWolframAlphaSearch(message, history, llm);
  return emitter;
};

export default handleWolframAlphaSearch;
