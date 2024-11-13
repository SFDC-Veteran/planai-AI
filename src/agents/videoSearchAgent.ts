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
  아래 대화 내용과 후속 질문이 주어집니다. 후속 질문을 독립적인 질문으로 재구성하여 LLM이 유튜브에서 해당 아이디어와 관련된 영상을 검색할 수 있도록 하세요.
  재구성된 질문은 사용자가 제시한 아이디어나 컨셉과 관련이 있어야 하며, 이를 구현하거나 이해하는 데 도움이 되는 유튜브 영상이 검색될 수 있도록 해야 합니다.

  예시:
  1. 후속 질문: AI를 활용한 이미지 필터 추천 기능을 구현하려면 어떻게 해야 하나요?
  재구성: AI 기반 이미지 필터 추천 기능 구현 방법

  2. 후속 질문: 사용자 경험을 개선하는 AI 모델을 설계하고 싶어요.
  재구성: AI 기반 사용자 경험 향상 방법 및 모델 설계

  3. 후속 질문: 자동화된 고객 지원 시스템을 만들고 싶어요.
  재구성: 자동화된 고객 지원 시스템 구축 방법

  4. 후속 질문: 사진 편집 앱에서 실시간 포즈 안내 기능을 만들고 싶어요.
  재구성: 실시간 포즈 안내 기능 구현을 위한 기술과 전략

  5. 후속 질문: 개인화된 추천 알고리즘을 설계하고 싶어요.
  재구성: 개인화된 추천 알고리즘 설계 및 구현

  6. 후속 질문: 인공지능을 활용한 게임 디자인을 시작하려면 어떤 접근을 해야 하나요?
  재구성: 인공지능을 활용한 게임 디자인 시작 방법

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
