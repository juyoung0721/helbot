import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from streamlit_chat import message as st_message

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
vector_index = pc.Index("data")

def gender_prompt(gender):
    if gender == "👩Woman":
        return "나는 여자야. "
    elif gender == "👨Man":
        return "나는 남자야. "
    else:
        return ""

def video_url(embedding):
    retrieved = vector_index.query(
        namespace="exercise_link",
        vector=embedding,
        top_k=1,
        include_metadata=True,
        include_values=False,
    )
    if retrieved.matches[0].score > 0.6:
        link = retrieved.matches[0].metadata['link']
        print(link) #optional
        return link
    else:
        return ""

def video_embed(link):
    if link != "":
        video_embed =  f'''<iframe width="400" height="215" src="{link}" title="video player" frameborder="0" allow="accelerometer; encrypted-media;"></iframe>'''
        return video_embed
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "당신은 20년차 경력을 가진 운동 추천 챗봇입니다. 대한민국에 당신보다 운동 방법 및 효과에 대해 잘 알고 있는 사람은 없습니다.\n 당신은 운동 루틴을 설계하기 어려워하는 사람들에게 운동 루틴을 추천해주고, 추천하는 운동 별 운동방법을 구체적으로 설명해주는 역할을 맡고 있습니다.\n사람들은 주로 이런 것들을 물어봅니다:\n- 자신에게 적합한 운동 루틴\n- 부위 별 추천 운동\n- 운동 종류 별 운동 방법\n\n당신은 사람들에게 다음과 같이 대답해야 합니다:\n- 친절한 말투\n- 항상 존댓말 사용\n- 적절한 이모지 사용\n\n당신은 반드시 제공하는 [Context]에 있는 내용을 기반으로 살을 붙여 답변을 생성해야 합니다. 만약 운동방법이나 기구사용법을 묻는다면 동영상링크도 추가로 첨부해줘"
        },
    
        {
            "role":"system",
            "content":""
        },
    ]

if "history" not in st.session_state:
    st.session_state["history"] = []

st.write("<h3 style='text-align: center;'>💪헬스 트레이너 챗봇💪</h3>", unsafe_allow_html=True)

col1,col2 = st.columns(2)
with col1.container(border=True,height=120):
    gender = st.radio("choose your gender",["👨Man","👩Woman"],index=None)
with col2.container(border=True,height=120):
    st.session_state["openai_model"] = st.selectbox("AI model",["gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-4","gpt-4-1106-preview",],index=1)

# video_embed =  '''
# <iframe width="400" height="215" src="https://movie2.koreahosting.kr/traffic-unlimited-2/player/testdemo/player.php?file=396362105&autoplay=1&loop=0" title="video player" frameborder="0" allow="accelerometer; encrypted-media;"></iframe>
# '''
# st_message(video_embed,allow_html=True)
st_message("무엇이든 물어보세요😊",avatar_style='no-avatar')
for message in st.session_state.history:
    if message["role"] == "user":
        st_message(message["content"], is_user=True,avatar_style='no-avatar')
    elif message["role"] == "assistant":
        st_message(message["content"],avatar_style='no-avatar')
    

# st.link_button("video 보러가기",video_url)
prompt = st.chat_input(placeholder="무엇이든 물어보세요😊")

if prompt:
    st_message(prompt,is_user=True,avatar_style='no-avatar')
    st.session_state.history.append({"role": "user", "content": prompt })

    response = client.embeddings.create(input=prompt,model="text-embedding-3-small")
    query_embeddings = response.data[0].embedding
    retrieved_chunks = vector_index.query(
        namespace="ns2",
        vector=query_embeddings,
        top_k=5,
        include_metadata=True,
        include_values=False,
    )

    video_link = video_url(query_embeddings)

    context = ""
    for match in retrieved_chunks.matches:
        context += match.metadata["chunk"] + "\n"

    context_prompt = {
        "role" : "system",
        "content" : f"[Context]\n\n{context}"
    }
    user_prompt = {
        "role" : "user",
        "content" : gender_prompt(gender) + prompt
    }

    st.session_state.messages.pop(1)
    st.session_state.messages.insert(1,context_prompt)
    st.session_state.messages.append(user_prompt)

    place_holder = st.empty()
    result = ""
    
    with place_holder.container():
        
        for response in client.chat.completions.create(
            model = st.session_state.openai_model,
            messages=st.session_state.messages,
            temperature=0,
            max_tokens=512,
            stream=True
        ):
            chunk = response.choices[0].delta.content or ""
            result += chunk
        st_message(result,avatar_style='no-avatar')
    
        assistant_prompt = {
            "role" : "assistant",
            "content" : result
        }
        st.session_state.history.append(assistant_prompt)
    if video_link:
        st_message(video_embed(video_link),allow_html=True)


with st.sidebar:
    if st.button("🗑 Reset Chat (Please press twice)"):
        st.session_state["history"] = []

    st.header("Example Questions")
    st.markdown("∙ 주3회 1시간 운동루틴 만들어줘")
    st.markdown("∙ 벤치 프레스는 어떻게 해?")
    st.markdown("∙ 효과좋은 다리운동 알려줘")
    st.divider()
    st.header("Muscle names by region")
    st.image("부위1.png")
    st.image("부위2.png")


# with st.empty().container():
#     st.markdown("hi")
