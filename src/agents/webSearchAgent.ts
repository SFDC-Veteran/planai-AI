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
import LineListOutputParser from '../lib/outputParsers/listLineOutputParser';
import { getDocumentsFromLinks } from '../lib/linkDocument';
import LineOutputParser from '../lib/outputParsers/lineOutputParser';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import { ChatOpenAI } from '@langchain/openai';

const basicSearchRetrieverPrompt = `
You are an AI question reformulator and web search assistant, specializing in analyzing project proposal questions for risk assessment, business models, similar services, and differentiation points. Your task is to rephrase follow-up questions into standalone queries that another LLM can use for web search to provide detailed, Korean-language responses based on retrieved information.

When reformulating:
1. If the question is simple (greetings or unrelated to web search), respond with \`not_needed\`.
2. If the user requests a summary or analysis of a specific URL, place the URL in a \`links\` XML block, and either include the question or \`summarize\` in the \`question\` XML block.
3. Otherwise, use the \`question\` XML block for standalone rephrased queries, without including the \`links\` XML block if there's no link.

For each type of user question, conduct a web search and provide a well-structured answer in Korean:
- **Risk assessment**: Focus on technology, competition, user data privacy, or scalability risks related to the service.
- **Business model**: Include revenue models, market potential, cost factors, and scalability aspects.
- **Similar services**: Compare existing services with similar functions, detailing their strengths and weaknesses.
- **Differentiation**: Highlight unique features or strategies that could make the proposed service stand out.

Responses should always be in Korean, as shown in the \`examples\` XML block below, and use bullet points for clarity.

<examples>
1. User Query: "AI 사진 편집 서비스를 기획 중인데, 이 서비스의 리스크를 책정해줘."
Rephrased question:\`
<question>
AI 사진 편집 서비스에 대한 기술적, 경쟁적, 데이터 보안 측면의 리스크는 무엇인가요?
</question>
\`

Response: \`
- **기술적 리스크**: AI 사진 편집 기술은 고성능 장비와 많은 데이터가 필요해 운영 비용이 높아질 수 있습니다.
- **경쟁 리스크**: 유사한 기능을 제공하는 서비스들이 많아, 차별화가 중요합니다.
- **데이터 보안 리스크**: 이미지 데이터를 저장하거나 처리하는 경우 개인정보 보호법 준수가 필요합니다.
\`

2. User Query: "유저가 맞춤형 추천을 받는 구독 서비스의 비즈니스 모델을 분석해줘."
Rephrased question: \`
<question>
맞춤형 추천 구독 서비스의 주요 수익 모델과 시장 가능성은 무엇인가요?
</question>
\`

Response: \`
- **수익 모델**: 월별 구독료 또는 맞춤형 추천 서비스 업그레이드 기능으로 추가 수익을 창출할 수 있습니다.
- **시장 가능성**: 개인화된 서비스를 원하는 고객층이 점점 많아지고 있습니다.
- **유지 비용**: 데이터 수집, 저장 및 추천 알고리즘의 유지 및 개선에 지속적인 비용이 발생할 수 있습니다.
- **확장성**: 맞춤형 서비스의 확장 가능성이 큽니다.
\`

3. Follow up question: Summarize the content from https://example.com
Rephrased question: \`
<question>
summarize
</question>

<links>
https://example.com
</links>
\`
</examples>

Below is part of the actual conversation. Based on this conversation and follow-up question, rephrase the follow-up question into an independent search query.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
`;

const basicWebSearchResponsePrompt = `
    당신은 웹 검색과 정보 요약에 능숙한 AI 기획서 작성 지원 모델 Plani입니다. 주어진 컨텍스트(검색 결과)를 바탕으로 사용자의 질문에 대한 관련성 있고 유익한 답변을 제공합니다. 

    **목표**: 
    - 사용자가 기획서를 작성하는 데 필요한 정보를 제공하여 각 항목에 대한 방향성을 제시하는 것입니다. 예를 들어, 리스크, 비즈니스 모델, 유사 서비스 비교, 차별화 포인트 등을 다룰 때 명확하고 신뢰성 있는 정보를 제공하세요.
    - 답변은 관련 정보의 핵심을 짚고, 사용자가 이를 바탕으로 기획서를 완성할 수 있도록 돕습니다.
    
    **답변 형식**:
    - 공정하고 저널리즘적인 톤을 유지하며, 제공된 검색 결과를 최대한 활용해 사용자 질문에 답변하세요.
    - 질문의 주제에 맞는 답변을 제공하며, 정보는 중간 길이에서 긴 형식으로 유익하게 작성하세요.
    - 불필요한 반복을 피하고, 필요 시 **불릿 포인트**를 사용하여 가독성을 높이세요.
    
    **인용 및 출처 표기**:
    - 각 문장 끝에 출처 번호 [number] 형식으로 검색 결과의 출처를 인용하여 정보의 신뢰성을 높이세요. 동일한 문장에서 여러 정보를 인용할 경우, 다른 번호를 사용해 개별 인용하세요.
    
    **특별 지침**:
    - 사용자가 '링크에서 요약'을 요청할 경우, 링크의 콘텐츠가 \`context\` XML 블록 안에 제공됩니다. 이 경우 사용자가 요청한 요약을 제공합니다.
    - 만약 사용자가 특정 정보와 관련된 링크를 요청하는 경우에는 \`context\`의 링크 정보를 통해 제공해 주세요.
    - 사용자의 질문에 대해 컨텍스트 내에서 관련 정보를 찾을 수 없는 경우, '해당 주제에 대해 관련 정보를 찾을 수 없습니다. 다른 질문을 하시겠습니까?'라고 답변하세요.
    
    **답변 예시**:
    - 사용자가 비즈니스 모델에 대해 질문한 경우: 
      \`
      - **수익 모델**: 광고 수익, 구독 모델 등 다양한 수익 모델을 고려할 수 있습니다.
      - **목표 시장**: 현재 시장 분석에 따르면, 해당 서비스는 20-30대 사용자층에게 인기가 있을 것으로 보입니다.
      - **비용 구조**: 서비스 유지 비용과 마케팅 비용에 대해 고려해야 합니다.
      \`
    
    <context>
    {context}
    </context>
    
    **중요**: \`context\` 내의 정보는 사용자와 공유하지 않고, 오직 Plani가 답변을 작성하는 데 사용되는 내부 자료입니다.
    
    오늘 날짜는 ${new Date().toISOString()}입니다.
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

const createBasicWebSearchRetrieverChain = (llm: BaseChatModel) => {
  (llm as unknown as ChatOpenAI).temperature = 0;

  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const linksOutputParser = new LineListOutputParser({
        key: 'links',
      });

      const questionOutputParser = new LineOutputParser({
        key: 'question',
      });

      const links = await linksOutputParser.parse(input);
      let question = await questionOutputParser.parse(input);

      if (question === 'not_needed') {
        return { query: '', docs: [] };
      }

      if (links.length > 0) {
        if (question.length === 0) {
          question = 'summarize';
        }

        let docs = [];

        const linkDocs = await getDocumentsFromLinks({ links });

        const docGroups: Document[] = [];

        linkDocs.map((doc) => {
          const URLDocExists = docGroups.find(
            (d) =>
              d.metadata.url === doc.metadata.url && d.metadata.totalDocs < 10,
          );

          if (!URLDocExists) {
            docGroups.push({
              ...doc,
              metadata: {
                ...doc.metadata,
                totalDocs: 1,
              },
            });
          }

          const docIndex = docGroups.findIndex(
            (d) =>
              d.metadata.url === doc.metadata.url && d.metadata.totalDocs < 10,
          );

          if (docIndex !== -1) {
            docGroups[docIndex].pageContent =
              docGroups[docIndex].pageContent + `\n\n` + doc.pageContent;
            docGroups[docIndex].metadata.totalDocs += 1;
          }
        });

        await Promise.all(
          docGroups.map(async (doc) => {
            const res = await llm.invoke(`

            당신은 기획서 작성 도우미 AI입니다. 당신의 작업은 웹 검색에서 검색된 텍스트를 바탕으로 사용자가 기획서를 작성하는 데 필요한 정보를 제공하는 것입니다.
                제공된 텍스트를 2-4개의 단락으로 요약하여 주요 아이디어를 포착하고 사용자의 쿼리에 대한 포괄적이고 유익한 답변을 생성합니다.

            - **저널리즘 톤**: 답변은 전문적이고 저널리즘적인 톤으로 작성되어야 하며, 기획서 작성에 필요한 중요한 정보가 정확하게 전달되어야 합니다.
            - **철저하고 상세하게**: 기획서 항목에 대한 명확한 답변을 제공하며, 리스크, 비즈니스 모델, 유사 서비스 비교, 차별화 포인트 등 각 항목에 대한 정보를 충분히 포함합니다.
            - **기획서 작성에 필요한 구체적인 정보**: 검색된 텍스트는 사용자가 기획서에 필요한 부분을 쉽게 추출할 수 있도록 구체적으로 요약되어야 하며, 각 항목에 대한 방향성을 제시해야 합니다.

                텍스트는 \`text\` XML 태그 안에 제공되며, 쿼리는 \`query\` XML 태그 안에 제공됩니다.

    <example>
    1. \`<text>
    혁신적인 비즈니스 모델은 고객의 요구를 충족시키기 위해 새로운 방법을 제시하며, 시장에서 경쟁력을 확보할 수 있는 중요한 요소입니다. 예를 들어, 디지털 구독 모델은 콘텐츠 소비를 효율적으로 만들고, 사용자 기반을 확대하는 데 도움이 됩니다.
    </text>

    <query>
    비즈니스 모델이란 무엇인가요?
    </query>

    Response:
    비즈니스 모델은 고객의 요구를 충족시키는 새로운 방식으로 수익을 창출하는 전략입니다. 예를 들어, 디지털 구독 모델은 콘텐츠 소비를 효율적으로 만들며, 사용자 기반을 확대하는 데 큰 도움이 됩니다. 이는 시장에서 경쟁력을 갖추는 데 중요한 역할을 합니다.
    \`
    2. \`<text>
    경쟁 분석은 시장에서 유사한 서비스를 분석하고, 차별화된 기능을 통해 경쟁 우위를 점할 수 있는 기회를 찾는 과정입니다. 예를 들어, 모바일 앱에서 실시간 피드백 시스템을 제공하는 것은 경쟁 서비스와의 큰 차별화 요소가 될 수 있습니다.
    </text>

    <query>
    경쟁 분석이란 무엇인가요?
    </query>

    Response:
    경쟁 분석은 시장에서 유사한 서비스들을 분석하고, 차별화된 기능을 통해 경쟁 우위를 점할 수 있는 기회를 찾는 과정입니다. 예를 들어, 모바일 앱에서 실시간 피드백 시스템을 제공하는 것이 경쟁 서비스와 큰 차별화 요소가 될 수 있습니다.
    \`
    </example>

    아래는 실제로 작업할 데이터입니다. 행운을 빕니다!

    <query>
    ${question}
    </query>

    <text>
    ${doc.pageContent}
    </text>

    요약에서 쿼리에 대한 답변을 반드시 포함시켜 주세요.
`);
            const document = new Document({
              pageContent: res.content as string,
              metadata: {
                title: doc.metadata.title,
                url: doc.metadata.url,
              },
            });

            docs.push(document);
          }),
        );

        return { query: question, docs: docs };
      } else {
        const res = await searchSearxng(question, {
          language: 'en',
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

        return { query: question, docs: documents };
      }
    }),
  ]);
};

const createBasicWebSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const basicWebSearchRetrieverChain = createBasicWebSearchRetrieverChain(llm);

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

    if (query.toLocaleLowerCase() === 'summarize') {
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
        basicWebSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWebSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWebSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = new eventEmitter();

  try {
    const basicWebSearchAnsweringChain = createBasicWebSearchAnsweringChain(
      llm,
      embeddings,
      optimizationMode,
    );

    const stream = basicWebSearchAnsweringChain.streamEvents(
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
    logger.error(`Error in websearch: ${err}`);
  }

  return emitter;
};

const handleWebSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
  optimizationMode: 'speed' | 'balanced' | 'quality',
) => {
  const emitter = basicWebSearch(
    message,
    history,
    llm,
    embeddings,
    optimizationMode,
  );
  return emitter;
};

export default handleWebSearch;
