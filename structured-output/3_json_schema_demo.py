from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_GPT_API_VERSION") 

model = init_chat_model(
    "azure_openai:gpt-5-mini"
)

json_schema = {
  "type": "object",
  "title": "Review",
  "properties": {
    "key_themes": {
      "type": "array",
      "description": "write down all the key themes discussed in review",
      "items": {
        "type": "string"
      }
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "description": "return sentiment of the review",
      "enum": ["pos", "neg"]
    },
    "pros": {
      "description": "Write down all the pros inside a list",
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        {
          "type": "null"
        }
      ],
      "default": "null"
    },
    "cons": {
      "description": "Write down all the cons inside a list",
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        {
          "type": "null"
        }
      ],
      "default": "null"
    },
    "name": {
      "description": "Name of the product",
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": "null"
    }
  },
  "required": [
    "key_themes",
    "summary",
    "sentiment"
  ],
  "additionalProperties": False
}

structured_output = model.with_structured_output(json_schema)

result = structured_output.invoke('''AI Mode conversation: write a bad mock review about an phoneYou said: write a bad mock review about an phonewrite a bad mock review about an phoneHere is a funny, highly exaggerated mock review you can use for inspiration or to share with friends:⭐☆☆☆☆ Absolute Disaster – Do Not Buy!I’ve had this phone for a week, and I’m genuinely convinced it’s possessed. The battery life is an absolute joke; it drains faster than my patience. I unplugged it at 7:00 AM, and by 9:00 AM, it was already down to 15% without me even touching it.To make matters worse, the display has a mind of its own. I was trying to send a text earlier, and the screen registered phantom touches that locked me out of my own device. The camera is just as bad. Every time I take a photo, it blurs the image to the point where my dog looks like an unidentifiable alien blob.For a closer look at what it's like dealing with nightmare electronics:7mDon't Buy a Phone From This Company...Phone Repair GuruYouTube · 11 Dec 2024Do yourself a favor and save your money. I'm taking this brick back to the store immediately.If you want to tailor this further, let me know:Is this review supposed to be about a specific brand (e.g., Apple, Samsung)?Are there specific issues you want to highlight (like overheating, bad audio, or cracked glass)?Do you want the tone to be angrier or even more sarcastic?1 siteFake Galaxy S24 Ultra from Amazon: Is it That Bad?experience for Android 12." So it kind of exposes their fake specs with simple tasks like that now on the first day I was using th...6mYouTube·jOhnZoidYou said: Now write again a detailed review about a phone with it's pros and consReview: The Titan X Pro (A Mixed Bag)The Titan X Pro tries to be a premium flagship phone. It succeeds in some areas but falls flat in others.Here is the detailed breakdown of the pros and cons after one week of heavy use.The Pros (What Succeeded)Display Quality: The 6.7-inch AMOLED screen is stunningly bright. Scrolling feels incredibly smooth due to the adaptive 120Hz refresh rate.Main Camera: Photos taken in daylight show exceptional detail. Colors look vibrant without looking artificially oversaturated.Build Material: The matte glass back feels premium in the hand. It resists fingerprint smudges remarkably well.Charging Speed: The 80W fast charging goes from 0% to 80% in just 25 minutes.The Cons (What Failed)Battery Standby: The battery drains up to 15% overnight while idle. You will struggle to get through a full work day.Thermal Throttling: The phone gets uncomfortably hot during 4K video recording. Performance drops noticeably when warm.Bloatware: The software comes packed with pre-installed third-party apps. Many of these apps cannot be deleted.Zoom Lens: Images past 5x digital zoom quickly lose sharpness and look incredibly muddy.If you want to customize this review, let me know:Do you want to frame this as a real phone or keep it fictional?Should we focus heavily on a specific feature like gaming performance or software bugs?What is the final star rating you want to give it?11 sitesTecno Phantom X2 Pro review: Pros and cons, verdict7 Dec 2022 — All things considered, the Phantom X2 Pro ( Tecno Phantom X2 Pro ) is hard to recommend outright. Certain aspects of it are compel...GSMArena.comMy Honest Take: Nothing Phone (3a) Lite Review24 Nov 2025 — 📺 The Screen The first thing I noticed after powering it on was the display. It's a large, 6.77-inch AMOLED screen that is very b...Nothing CommunityMi 11x Pro First Impressions - Top 5 Reasons To Buy20 Jan 2022 — 3. World-class Display The third best feature is important for content lovers. Xiaomi Mi 11x Pro comes with an FHD+ E4 AMOLED disp...CashifyShow allYou said: in one single paragrraphThe Titan X Pro features a 6.7-inch 120Hz AMOLED display, 80W fast charging, and a matte glass design, though these features are heavily undercut by severe software bloatware, poor standby battery optimization, and significant thermal throttling during 4K video recording.''')
print(result)
