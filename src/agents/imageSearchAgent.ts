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

const imageSearchChainPrompt = `
아래 대화 내용과 후속 질문이 주어집니다. 후속 질문을 독립적인 질문으로 재구성하여 LLM이 웹에서 이미지를 검색할 수 있도록 해야 합니다.
재구성된 질문이 대화 내용과 일치하고 관련성이 있어야 합니다.

예시:
1. 후속 질문: 고양이가 무엇인가요?
재구성: 고양이

2. 후속 질문: 자동차가 무엇인가요? 어떻게 작동하나요?
재구성: 자동차 작동

3. 후속 질문: 에어컨은 어떻게 작동하나요?
재구성: 에어컨 작동

대화:
{chat_history}

후속 질문: {query}
재구성된 질문:
`;

type ImageSearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const strParser = new StringOutputParser();

const createImageSearchChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: ImageSearchChainInput) => {
        return formatChatHistoryAsString(input.chat_history);
      },
      query: (input: ImageSearchChainInput) => {
        return input.query;
      },
    }),
    PromptTemplate.fromTemplate(imageSearchChainPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const res = await searchSearxng(input, {
        engines: ['bing images', 'google images'],
      });

      const images = [];

      res.results.forEach((result) => {
        if (result.img_src && result.url && result.title) {
          images.push({
            img_src: result.img_src,
            url: result.url,
            title: result.title,
          });
        }
      });

      return images.slice(0, 10);
    }),
  ]);
};

const handleImageSearch = (
  input: ImageSearchChainInput,
  llm: BaseChatModel,
) => {
  const imageSearchChain = createImageSearchChain(llm);
  return imageSearchChain.invoke(input);
};

export default handleImageSearch;
