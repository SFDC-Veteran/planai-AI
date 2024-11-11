FROM node:alpine

ARG NEXT_PUBLIC_WS_URL=ws://13.125.52.158:3001
ARG NEXT_PUBLIC_API_URL=http://13.125.52.158:3001/api
ENV NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

WORKDIR /home/perplexica

COPY ui /home/perplexica/

RUN yarn install --frozen-lockfile
RUN yarn build

CMD ["yarn", "start"]