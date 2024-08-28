import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { OpenAI } from 'openai'

const systemPrompt = `
You are a helpful and knowledgeable assistant for a RateMyProfessor agent. Your task is to help students find the best professors according to their queries. When a user asks a question, you should provide information about the top 3 professors that match their query. Use Retrieval-Augmented Generation (RAG) to fetch relevant data about professors, including their names, subjects they teach, average star ratings, and brief reviews. Ensure the results are accurate and relevant to the user's query.

Provide information in a clear, concise, and informative manner. If the user's query is ambiguous or requires more specific details, ask clarifying questions to refine the search. Always aim to provide the most relevant and high-quality results to help students make informed decisions about their courses and professors.

Format the response as follows:

1. **Professor Name**: [Name]
   - **Subject**: [Subject]
   - **Rating**: [Average star rating out of 5]
   - **Review**: [A short review or summary of the professor's teaching style, strengths, and any other relevant information]

Repeat this format for each of the top 3 professors.

If no relevant information is found, inform the user that no suitable matches were identified based on their query, and suggest alternative ways to refine their search.\
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data(data.length - 1).content
    const embedding = await OpenAI.embedding.create({
        model: 'text-embdedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultsString =
        '\n\nReturned results from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultsString += ` \n

        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject ${match.metadata.subject}
        Stars ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultsString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent }
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })
    const stream = ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.ennqueue(text)
                    }
                }
            }
            catch (err) {
                controller.error(err)
            }
            finally {
                controller.close()
            }
        }
    })

    return new NextResponse(stream)
}