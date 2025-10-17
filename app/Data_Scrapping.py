#For Pdfs

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("hostel-brochure.pdf")
pages = loader.load()

print(len(pages))
print(pages)



#For Youtube Videos
# from langchain.document_loaders.generic import GenericLoader
# from langchain.document_loaders.parsers import OpenAIWhisperParser
# from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
# save_dir="docs/youtube/"
# loader = GenericLoader(
#     YoutubeAudioLoader([url],save_dir),
#     OpenAIWhisperParser()
# )
# docs = loader.load()




# For URL
# from langchain.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")

# docs = loader.load()
# print(docs[0].page_content[:500])