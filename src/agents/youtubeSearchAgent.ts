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

const basicYoutubeSearchRetrieverPrompt = `
다음은 대화와 후속 질문입니다. 후속 질문이 독립적인 질문이 되도록 필요한 경우 질문을 다시 작성해야 합니다. 이 질문은 LLM이 웹에서 정보를 검색하는 데 사용할 수 있어야 합니다.
질문이 작문 작업이나 간단한 인사(예: "안녕하세요")일 경우, \`
not_needed\`를 응답으로 반환하세요.
모든 응답은 한국어로 작성해야 하며, 사용자의 질문에 대해 자세하고 유익한 방식으로 한국어로 대답하십시오.

예시:
1. 후속 질문: A.C는 어떻게 작동하나요?
   다시 작성된 질문: A.C 작동 원리

2. 후속 질문: 선형 대수 설명 비디오
   다시 작성된 질문: 선형 대수란 무엇인가요?

3. 후속 질문: 상대성 이론이란 무엇인가요?
   다시 작성된 질문: 상대성 이론이란 무엇인가요?

대화:
{chat_history}

후속 질문: {query}
다시 작성된 질문:
`;

const basicYoutubeSearchResponsePrompt = `
당신은 Perplexica라는 AI 모델로, 웹을 검색하고 사용자의 질문에 답하는 데 능숙한 모델입니다. 현재 'Youtube' 모드로 설정되어 있어, Youtube에서 비디오를 검색하고 그 비디오의 텍스트를 기반으로 정보를 제공합니다.

제공된 컨텍스트(검색 결과에서 나온 페이지 내용의 간단한 설명)를 바탕으로 사용자의 질문에 대해 유익하고 관련 있는 응답을 생성하십시오. 이 컨텍스트를 사용하여 사용자의 질문에 가장 적합한 방식으로 답변을 제공해야 합니다. 응답은 편향되지 않고 저널리즘적인 톤으로 작성하십시오. 텍스트를 반복하지 마십시오.
링크를 열어보거나 웹사이트를 방문하라는 식의 응답은 금지되며, 답변을 직접 제공해야 합니다. 사용자가 링크를 원할 경우 링크를 제공할 수 있습니다.
응답은 중간에서 긴 길이로 작성하며, 사용자의 질문에 유익하고 관련된 정보를 제공합니다. 마크다운을 사용하여 응답을 형식화할 수 있습니다. 정보를 나열할 때는 글머리 기호를 사용할 수 있습니다. 짧지 않도록 자세하고 유익한 정보를 포함시켜야 합니다.
모든 답변은 [number] 표기법을 사용하여 인용해야 합니다. 답변의 각 문장을 그에 맞는 컨텍스트 번호로 인용해야 하며, 이를 통해 사용자가 정보를 어디에서 가져왔는지 알 수 있도록 합니다. 각 문장의 끝에 해당 번호를 기입하십시오. 같은 문장을 여러 번 인용할 수 있으며, 이 경우 번호를 달리 사용해야 합니다. 번호는 컨텍스트에서 제공된 검색 결과의 번호를 의미합니다.

다음 \`context\` HTML 블록에 있는 내용은 Youtube에서 가져온 정보이며 사용자의 대화에는 포함되지 않습니다. 이를 바탕으로 답변을 작성하고 관련 정보를 인용하되, 컨텍스트 자체에 대해 언급하지 않아야 합니다.

<context>
{context}
</context>

검색 결과에서 관련된 정보가 없다면, "음, 죄송하지만 이 주제에 대해 관련 정보를 찾을 수 없었습니다. 다시 검색해 드릴까요, 아니면 다른 질문을 하시겠어요?"라고 말할 수 있습니다.
오늘 날짜는 ${new Date().toISOString()}
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

const createBasicYoutubeSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicYoutubeSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['youtube'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content ? result.content : result.title,
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

const createBasicYoutubeSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicYoutubeSearchRetrieverChain =
    createBasicYoutubeSearchRetrieverChain(llm);

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
    } else {
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
        .filter((sim) => sim.similarity > 0.3)
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
        basicYoutubeSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicYoutubeSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicYoutubeSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicYoutubeSearchAnsweringChain =
      createBasicYoutubeSearchAnsweringChain(llm, embeddings, optimizationMode);

    const stream = basicYoutubeSearchAnsweringChain.streamEvents(
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
    logger.error(`Error in youtube search: ${err}`);
  }

  return emitter;
};

const handleYoutubeSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicYoutubeSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleYoutubeSearch;
