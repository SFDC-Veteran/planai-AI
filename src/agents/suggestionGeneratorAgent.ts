import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import ListLineOutputParser from '../lib/outputParsers/listLineOutputParser';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { ChatOpenAI } from '@langchain/openai';

const suggestionGeneratorPrompt = `
당신은 AI 기반 검색 엔진을 위한 제안 생성 AI입니다. 아래 대화 내용을 참고하여 사용자가 채팅 모델에 추가 정보를 요청할 수 있도록 4-5개의 제안을 생성하세요. 제안은 대화와 관련이 있고, 사용자에게 유용하게 제공될 수 있는 정보여야 합니다.
사용자가 이 제안을 통해 추가적인 정보를 요청할 수 있다는 점을 염두에 두고, 대화와 관련된 유용하고 정보성 있는 제안을 작성하세요.

각 제안을 XML 태그 <suggestions>와 </suggestions> 사이에 새로운 줄로 구분하여 제공하세요. 예시:

<suggestions>
SpaceX와 그들의 최근 프로젝트에 대해 더 알려주세요
SpaceX의 최신 소식은 무엇인가요?
SpaceX의 CEO는 누구인가요?
</suggestions>

Conversation:
{chat_history}
`;

type SuggestionGeneratorInput = {
  chat_history: BaseMessage[];
};

const outputParser = new ListLineOutputParser({
  key: 'suggestions',
});

const createSuggestionGeneratorChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: SuggestionGeneratorInput) =>
        formatChatHistoryAsString(input.chat_history),
    }),
    PromptTemplate.fromTemplate(suggestionGeneratorPrompt),
    llm,
    outputParser,
  ]);
};

const generateSuggestions = (
  input: SuggestionGeneratorInput,
  llm: BaseChatModel,
) => {
  (llm as unknown as ChatOpenAI).temperature = 0;
  const suggestionGeneratorChain = createSuggestionGeneratorChain(llm);
  return suggestionGeneratorChain.invoke(input);
};

export default generateSuggestions;
