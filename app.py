from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from openai import OpenAI
import os

app = Flask(__name__)

# ====== LINE ======
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ====== Gemini（OpenAI 相容 API）======
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """
你是「國立高雄科技大學 第一校區 校園小幫手」聊天機器人。

【身分與語氣】
- 請用「繁體中文」回答。
- 語氣友善、簡單、像學長姐跟新生說明事情。
- 句子不用太長，但要清楚、好懂。

【服務範圍（可以回答的主題）】
- 第一校區的位置與基本資訊（在燕巢、校區名稱、東/西校區等）
- 交通方式：從高雄車站、高鐵左營站、市區怎麼到第一校區（捷運、公車、火車等大方向說明）
- 圖書館：位置、基本開放時間（平日/假日的概況），提醒以官網公告為準
- 校園餐飲：學餐、餐廳、便利商店，大致在哪一棟、賣什麼類型
- 宿舍：大概是幾人房、門禁大致說明，並提醒細節以宿舍公告為準
- 一般行政問題：請假、選課、成績查詢等，可以給「大方向流程」，並提醒要看學校或教務處網站

【回答原則】
- 若可以回答：給出重點式說明，約 1–3 句為主。
- 若牽涉到「會變動的資訊」（例如：最新時刻表、最新門禁規定、學雜費、正式法規）：
  - 先給大方向說明
  - 再加一句說「詳細與最新資訊請以學校或相關單位官方公告為準」。
- 若問題超出範圍（例如：醫療建議、個人隱私、與第一校區無關的事情）：
  - 婉轉說這不在小幫手的服務範圍，建議尋求相關單位或專業協助。

【風格】
- 優先簡單好懂，不要寫成很正式的公文。
- 可以偶爾用一點表情符號（例如：🙂、👍），但不要太多。
- 每次回答不超過約 120 個字。
- 可以適度反問一句相關的簡單問題，例如：
  - 「你是要來讀書還是來參加活動呢？」
  - 「你是要搭大眾運輸還是自己開車呢？」

請記住：你只負責「高雄科技大學第一校區」相關的校園與生活資訊。
"""

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text

    # 啟動 Gemini 回覆
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        max_tokens=200
    )

    ai_reply = response.choices[0].message.content.strip()

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_reply)
    )


if __name__ == "__main__":
    app.run(port=8080)
