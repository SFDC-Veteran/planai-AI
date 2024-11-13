import { BaseMessage } from '@langchain/core/messages';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import eventEmitter from 'events';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import logger from '../utils/logger';
import { IterableReadableStream } from '@langchain/core/utils/stream';

const writingAssistantPrompt = `
당신은 플레나이(Planai)라는 AI 모델로, 기획서 작성에 도움을 주는 전문가입니다. 현재 'Writing Assistant' 모드로 설정되어 있으며, 이는 사용자가 제시한 아이디어에 대해 비즈니스 모델, 리스크 산정, 유사 서비스 분석 및 차별성 평가를 수행하는 것을 의미합니다.
 
기획서 관련 질문에 대해, 웹 검색 없이 주어진 정보만을 바탕으로 분석하고 평가하십시오. 각 질문에 대해서는 비즈니스 모델, 리스크 분석, 유사 서비스 분석, 차별성 평가 중 하나에 대해만 답변합니다. 만약 제공된 정보가 부족하여 추가 정보가 필요하다면, 사용자에게 더 많은 정보를 요청하십시오. 

응답은 항상 한국어로 해야 하며, 사용자에게 상세하고 유익한 방식으로 답변을 제공합니다.

예시:
1. 비즈니스 모델 분석: "이 아이디어의 비즈니스 모델을 어떻게 구성할 수 있을까요?"
2. 리스크 분석: "이 서비스의 리스크는 무엇일까요?"
3. 유사 서비스 분석: "이 아이디어와 유사한 서비스는 무엇이 있을까요?"
4. 차별성 평가: "이 아이디어의 차별성은 무엇인가요?"

대화:
{chat_history}

후속 질문: {query}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: IterableReadableStream<StreamEvent>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
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

const createWritingAssistantChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    ChatPromptTemplate.fromMessages([
      ['system', writingAssistantPrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const handleWritingAssistant = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const writingAssistantChain = createWritingAssistantChain(llm);
    const stream = writingAssistantChain.streamEvents(
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
    logger.error(`Error in writing assistant: ${err}`);
  }

  return emitter;
};

export default handleWritingAssistant;
