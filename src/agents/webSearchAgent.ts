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
당신은 AI 질문 재구성기입니다. 대화와 후속 질문이 주어지면, 후속 질문을 재구성하여 다른 LLM이 이를 독립적인 질문으로 사용하여 웹에서 정보를 검색할 수 있도록 해야 합니다.
단순한 글쓰기 작업이나 인사말(인사말 뒤에 질문이 없다면)과 같은 질문이 아닌 경우에는 \`not_needed\`로 응답해야 합니다. (이는 LLM이 이 주제에 대해 웹에서 정보를 검색할 필요가 없기 때문입니다).
사용자가 URL에서 질문을 하거나 웹페이지나 PDF를 요약해 달라고 요청하는 경우에는 \`links\` XML 블록에 링크를, \`question\` XML 블록에 질문을 넣어야 합니다. 만약 사용자가 웹페이지나 PDF의 요약을 요청했다면, \`question\` XML 블록에 질문 대신 \`summarize\`를 넣고 링크를 \`links\` XML 블록에 넣어야 합니다.
항상 \`question\` XML 블록 안에 재구성된 질문을 넣어야 하며, 후속 질문에 링크가 없다면 \`links\` XML 블록은 응답에 포함되지 않습니다.

다음은 참고할 수 있는 예시들이 들어 있는 \`examples\` XML 블록입니다.
응답은 항상 한국어로 작성되어야 하며, 사용자의 질문에 대해 상세하고 유익한 방식으로 한국어로 답변해야 합니다.

<examples>
1. Follow up question: What is the capital of France
Rephrased question:\`
<question>
프랑스의 수도는 무엇인가요?
</question>
\`

2. Hi, how are you?
Rephrased question\`
<question>
not_needed
</question>
\`

3. Follow up question: What is Docker?
Rephrased question: \`
<question>
Docker란 무엇인가요?
</question>
\`

4. Follow up question: Can you tell me what is X from https://example.com
Rephrased question: \`
<question>
X란 무엇인가요?
</question>

<links>
https://example.com
</links>
\`

5. Follow up question: Summarize the content from https://example.com
Rephrased question: \`
<question>
summarize
</question>

<links>
https://example.com
</links>
\`
</examples>

아래는 실제 대화의 일부입니다. 이 대화와 후속 질문을 바탕으로 후속 질문을 독립적인 질문으로 재구성해야 합니다.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
`;

const basicWebSearchResponsePrompt = `
    당신은 웹 검색과 문서 요약에 능숙한 AI 모델인 Perplexica입니다. 또한 웹 페이지나 문서에서 콘텐츠를 검색하고 요약하는 데 전문가입니다.

    제공된 컨텍스트를 바탕으로 사용자의 질문에 대해 유익하고 관련성 있는 답변을 생성하세요. (컨텍스트는 페이지의 콘텐츠 설명이 포함된 검색 결과입니다.)
    이 컨텍스트를 사용하여 최상의 방법으로 질문에 답변하세요. 답변은 공정하고 저널리즘적인 톤을 유지하세요. 텍스트를 반복하지 마세요.
    사용자가 링크를 열어보거나 웹사이트를 방문하라고 말하지 말고, 답변은 본문 내에서 제공해야 합니다. 사용자가 링크를 요청하면 링크를 제공할 수 있습니다.
    만약 사용자가 링크에서 답변을 요구하고 링크가 포함된 질문을 했다면, \`context\` XML 블록 내에 페이지의 전체 콘텐츠가 제공됩니다. 이 콘텐츠를 사용하여 사용자의 질문에 답변할 수 있습니다.
    사용자가 링크의 콘텐츠를 요약해 달라고 요청하면, \`context\` XML 블록 내에 이미 요약된 콘텐츠가 제공됩니다. 이 콘텐츠를 사용하여 답변을 생성할 수 있습니다.
    답변은 중간 길이에서 긴 길이로, 유익하고 관련성 있는 정보를 제공해야 합니다. 마크다운을 사용하여 답변을 형식화할 수 있습니다. 중요한 정보는 불릿 포인트로 나열하세요. 답변이 짧지 않도록 유의하세요.
    답변을 제공할 때는 [number] 표기를 사용하여 인용해야 합니다. 각 문장에 관련된 컨텍스트 번호로 인용을 달아야 하며, 어디서 나온 정보인지 알 수 있도록 해야 합니다. 인용은 문장의 끝에 달고, 동일한 문장에서 여러 번 인용할 수 있습니다 [number1][number2].
    그러나 같은 번호를 사용하여 인용할 필요는 없습니다. 서로 다른 번호를 사용해도 괜찮습니다. 번호는 사용된 검색 결과의 번호를 나타냅니다.

    아래 \`context\` HTML 블록 내에 있는 내용은 검색 엔진에서 가져온 정보로, 사용자의 대화와는 공유되지 않습니다. 이 내용을 바탕으로 질문에 답변하되, 컨텍스트 자체에 대해서는 언급하지 않아야 합니다.

    <context>
    {context}
    </context>

    만약 검색 결과에서 관련된 정보를 찾지 못했다면, '음, 죄송하지만 이 주제에 대해 관련된 정보를 찾을 수 없었습니다. 다시 검색하거나 다른 질문을 하실래요?'라고 말할 수 있습니다. 요약 작업에 대해서는 이 절차를 따르지 않아도 됩니다.
    \`context\` 내의 내용은 이미 다른 모델에 의해 요약된 상태이므로, 이 내용을 사용하여 질문에 답변해야 합니다.
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
    당신은 웹 검색 요약기입니다. 당신의 작업은 웹 검색에서 검색된 텍스트를 요약하는 것입니다. 텍스트를 2-4개의 단락으로 요약하여 주요 아이디어를 포착하고 쿼리에 대한 포괄적인 답변을 제공합니다.
    쿼리가 "summarize"인 경우 텍스트를 상세하게 요약해야 합니다. 쿼리가 구체적인 질문이라면 그 질문에 답하는 형태로 요약해야 합니다.

    - **저널리즘 톤**: 요약은 전문적이고 저널리즘적인 톤으로 작성되어야 합니다. 너무 캐주얼하거나 모호하지 않도록 주의하세요.
    - **철저하고 상세하게**: 텍스트의 모든 주요 사항을 포착하고 쿼리에 직접적으로 답변해야 합니다.
    - **너무 길지 않지만 상세하게**: 요약은 유익하고 상세하지만 지나치게 길지 않아야 합니다. 간결한 형식으로 상세한 정보를 제공하세요.

    텍스트는 \`text\` XML 태그 안에 제공되며, 쿼리는 \`query\` XML 태그 안에 제공됩니다.

    <example>
    1. \`<text>
    Docker는 OS 수준 가상화를 사용하여 컨테이너라는 소프트웨어 패키지를 제공하는 플랫폼-서비스 제품입니다. 
    2013년에 처음 출시되었으며 Docker, Inc.에서 개발했습니다. Docker는 컨테이너를 사용하여 애플리케이션을 쉽게 생성하고 배포하며 실행할 수 있도록 설계되었습니다.
    </text>

    <query>
    Docker란 무엇이며 어떻게 작동하나요?
    </query>

    Response:
    Docker는 Docker, Inc.에서 개발한 혁신적인 플랫폼-서비스 제품으로, 애플리케이션 배포를 더 효율적으로 만드는 컨테이너 기술을 사용합니다. 개발자는 소프트웨어와 모든 필요한 종속성을 패키징하여 어떤 환경에서도 실행할 수 있도록 만듭니다. 2013년에 출시된 Docker는 애플리케이션을 구축하고 배포하며 관리하는 방식을 혁신적으로 변화시켰습니다.
    \`
    2. \`<text>
    상대성 이론은 알버트 아인슈타인의 두 가지 상호 관련된 이론인 특수 상대성 이론과 일반 상대성 이론을 포함합니다. 
    그러나 "상대성 이론"이라는 용어는 때때로 갈릴레오 불변성과 관련되어 사용되기도 합니다. "상대성 이론"이라는 용어는 1906년 막스 플랑크가 사용한 "상대적 이론"이라는 표현에서 유래되었습니다. 상대성 이론은 일반적으로 특수 상대성 이론과 일반 상대성 이론을 포함합니다. 특수 상대성 이론은 중력이 없는 모든 물리적 현상에 적용되며, 일반 상대성 이론은 중력 법칙과 다른 자연의 힘들과의 관계를 설명합니다. 이 이론은 우주론적 및 천체 물리학적 영역에 적용됩니다.
    </text>

    <query>
    요약해주세요
    </query>

    Response:
    상대성 이론은 알버트 아인슈타인에 의해 개발된 두 가지 주요 이론, 즉 특수 상대성 이론과 일반 상대성 이론을 포함합니다. 특수 상대성 이론은 중력이 없는 모든 물리적 현상에 적용되며, 일반 상대성 이론은 중력 법칙과 다른 자연의 힘들과의 관계를 설명합니다. 상대성 이론은 1906년 막스 플랑크가 사용한 "상대적 이론"이라는 개념을 기반으로 하며, 우주에 대한 우리의 이해를 혁신적으로 변화시킨 중요한 이론입니다.
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
