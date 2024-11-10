import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { searchSearxng } from '../lib/searxng';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

const VideoSearchChainPrompt = `
  아래 대화 내용과 후속 질문이 주어집니다. 후속 질문을 독립적인 질문으로 재구성하여 LLM이 유튜브에서 관련 비디오를 검색할 수 있도록 하세요.
  재구성된 질문이 대화 내용과 일치하며 관련성이 있도록 해야 합니다.
  
  예시:
  1. 후속 질문: 자동차는 어떻게 작동하나요?
  재구성: 자동차 작동 원리
  
  2. 후속 질문: 상대성 이론이 무엇인가요?
  재구성: 상대성 이론이란 무엇인가
  
  3. 후속 질문: 에어컨은 어떻게 작동하나요?
  재구성: 에어컨 작동 원리
  
  Conversation:
  {chat_history}
  
  후속 질문: {query}
  재구성된 질문:
  `;

type VideoSearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const strParser = new StringOutputParser();

const createVideoSearchChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: VideoSearchChainInput) => {
        return formatChatHistoryAsString(input.chat_history);
      },
      query: (input: VideoSearchChainInput) => {
        return input.query;
      },
    }),
    PromptTemplate.fromTemplate(VideoSearchChainPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const res = await searchSearxng(input, {
        engines: ['youtube'],
      });

      const videos = [];

      res.results.forEach((result) => {
        if (
          result.thumbnail &&
          result.url &&
          result.title &&
          result.iframe_src
        ) {
          videos.push({
            img_src: result.thumbnail,
            url: result.url,
            title: result.title,
            iframe_src: result.iframe_src,
          });
        }
      });

      return videos.slice(0, 10);
    }),
  ]);
};

const handleVideoSearch = (
  input: VideoSearchChainInput,
  llm: BaseChatModel,
) => {
  const VideoSearchChain = createVideoSearchChain(llm);
  return VideoSearchChain.invoke(input);
};

export default handleVideoSearch;
