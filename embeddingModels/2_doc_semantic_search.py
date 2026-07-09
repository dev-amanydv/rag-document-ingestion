from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

embedding = AzureOpenAIEmbeddings(model="text-embedding-3-small")

documents = cricketers = [
    "Virat Kohli is an Indian batter known for his aggressive mindset, exceptional run-chasing ability, and elegant cover drives. He has captained India in all formats and is regarded as one of the greatest modern batsmen.",

    "Rohit Sharma is India's opening batter famous for his effortless timing and record-breaking ODI double centuries. He is nicknamed the Hitman and has led India in multiple ICC tournaments.",

    "Jasprit Bumrah is an Indian fast bowler recognized for his unique bowling action, deadly yorkers, and calm performances under pressure. He is considered one of the best death-over specialists in world cricket.",

    "MS Dhoni is a legendary Indian wicketkeeper-batter admired for his finishing ability, lightning-fast stumpings, and calm leadership. He captained India to victories in the T20 World Cup, ODI World Cup, and Champions Trophy.",

    "Sachin Tendulkar is widely known as the Master Blaster and is regarded as one of the greatest batsmen in cricket history. He scored 100 international centuries during his remarkable career.",

    "AB de Villiers from South Africa earned the nickname Mr. 360 because he could play innovative shots all around the ground. He was equally effective against both pace and spin bowling.",

    "Kane Williamson is the captain of New Zealand and is respected for his composure, consistency, and technically sound batting. He has led his team to multiple ICC tournament finals.",

    "Joe Root is an English batter known for his textbook technique and ability to score runs in all conditions. He has accumulated thousands of Test runs and remains a key player for England.",

    "Steve Smith from Australia is famous for his unconventional batting technique and incredible consistency in Test cricket. He has been one of the highest-ranked Test batsmen for several years.",

    "Pat Cummins is an Australian fast bowler and captain who combines pace, accuracy, and leadership. He has played a major role in Australia's success across all formats.",

    "Mitchell Starc is an Australian left-arm fast bowler renowned for his swinging deliveries and lethal yorkers. He has consistently been among the leading wicket-takers in ICC tournaments.",

    "Ben Stokes is an English all-rounder celebrated for his match-winning performances with both bat and ball. His innings in the 2019 World Cup final and the Headingley Ashes Test are considered legendary.",
    
    "Babar Azam is Pakistan's premier batter known for his elegant stroke play and consistency across formats. He has frequently ranked among the world's top ODI and T20 batters.",

    "Shaheen Shah Afridi is a Pakistani left-arm fast bowler who is famous for dismissing top-order batsmen with the new ball. His pace and swing make him a dangerous opening bowler.",

    "Rashid Khan from Afghanistan is a world-class leg-spinner known for his quick arm speed, deceptive googlies, and excellent T20 performances in leagues around the world.",

    "Jos Buttler is an English wicketkeeper-batter recognized for his explosive batting in limited-overs cricket. He is one of the most destructive finishers in T20 cricket.",

    "Trent Boult is a New Zealand left-arm fast bowler who excels at swinging the new ball and taking early wickets. He has been a key performer in both international cricket and franchise leagues.",

    "Lasith Malinga from Sri Lanka became famous for his slingy bowling action and deadly toe-crushing yorkers. He is one of the greatest T20 bowlers of all time.",

    "Chris Gayle from the West Indies is known as the Universe Boss for his powerful six-hitting ability. He has scored centuries in Tests, ODIs, and T20 Internationals and dominated T20 leagues worldwide.",

    "Sunil Narine from the West Indies is a mystery spinner who has also become an aggressive opening batter in T20 cricket. His all-round contributions have made him a valuable franchise player."
]

query = "Player known for cover drives"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(f"similaity: { documents[index]}, score: {score}")

