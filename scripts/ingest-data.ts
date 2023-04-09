import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { pinecone } from '@/utils/pinecone-client';
import { PDFLoader } from 'langchain/document_loaders';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import * as fs from 'fs';
import * as path from 'path';
/* Name of directory to retrieve files from. You can change this as required */
const filePath = 'docs/';

export const run = async (docPath: string) => {
  try {
    /*load raw docs from the pdf file */
    const loader = new PDFLoader(docPath);
    const rawDocs = await loader.load();

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    console.log(`split docs from ${docPath}`, docs);

    console.log('creating vector store...');
    /*create and store the embeddings in the vectorStore*/
    const embeddings = new OpenAIEmbeddings();
    const index = await pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name

    /* Pinecone recommends a limit of 100 vectors per upsert request to avoid errors*/
    const chunkSize = 50;
    for (let i = 0; i < docs.length; i += chunkSize) {
      const chunk = docs.slice(i, i + chunkSize);
      await PineconeStore.fromDocuments(
        index,
        chunk,
        embeddings,
        'text',
        PINECONE_NAME_SPACE,
      );
    }
  } catch (error) {
    console.log(`error processing ${docPath}`, error);
    throw new Error(`Failed to ingest ${docPath}`);
  }
};

(async () => {
  try {
    const files = fs.readdirSync(filePath);
    for (const file of files) {
      const docPath = path.join(filePath, file);

      if(file !== '.DS_Store'){
        await run(docPath);
      }


    }
    console.log('ingestion complete');
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }
})();
